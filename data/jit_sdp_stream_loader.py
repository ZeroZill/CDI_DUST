import warnings
from typing import Tuple, Union

import numpy as np
import torch


class JITSDPStreamLoader:
    r"""
    Initializes the JITSDPStreamLoader object for Just-In-Time Software Defect Prediction (JIT-SDP) data.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Feature vectors.
    Y : np.ndarray or torch.Tensor
        True labels for each commit (0: clean, 1: defect-inducing).
        Y may be provided as scalar labels or as one-hot encoded vectors.
    commit_unix_timestamp : np.ndarray
        Commit timestamps in milliseconds, ordered by commit.
    delays : np.ndarray
        Delay in days. For samples with label 1, indicates the delay (in days) before the fix information arrives.
    waiting_time : float
        Waiting time in days. For label 0 samples, the event time is calculated as:
            event_time = commit_time_in_days + waiting_time.
        For label 1 samples:
            event_time = commit_time_in_days + delay.
            (This implementation schedules one event per sample.)
    n_pretrain : int or None, optional (default=None)
        Number of pretraining samples. Pretraining samples are returned with their true labels directly.
    """

    def __init__(self,
                 X: Union[np.ndarray, torch.Tensor],
                 Y: Union[np.ndarray, torch.Tensor],
                 commit_unix_timestamp: np.ndarray,
                 delays: np.ndarray,
                 waiting_time: float = 15,
                 n_pretrain: Union[int, None] = None):
        # Assert equal lengths.
        assert len(X) == len(Y) == len(commit_unix_timestamp) == len(delays), \
            "X, Y, commit_unix_timestamp, and delays must have the same length."

        # Determine if the inputs are torch Tensors.
        self._is_tensor = isinstance(X, torch.Tensor)

        # Pretraining samples.
        self._next_is_pretrain = False
        self._n_pretrain = 0
        self._initial_X, self._initial_Y = None, None
        if n_pretrain is not None and n_pretrain > 0:
            self._next_is_pretrain = True
            self._n_pretrain = min(n_pretrain, len(Y))
            self._initial_X = X[:self._n_pretrain]
            self._initial_Y = Y[:self._n_pretrain]

        # Online data: remove the pretraining samples.
        self._online_X = X[self._n_pretrain:]
        self._online_Y = Y[self._n_pretrain:]
        self._online_commit_ts = commit_unix_timestamp[self._n_pretrain:]
        self._delays = delays[self._n_pretrain:]
        self._n_online = len(self._online_Y)

        # Test data time steps: 1,2,...,n_online.
        self._time_steps = np.arange(1, self._n_online + 1)
        self.waiting_time = waiting_time

        # Convert commit timestamps from ms to days (floating point).
        self._commit_days = self._online_commit_ts / 86400.0

        # Simulation time is based solely on online test data.
        self._t = 1
        self._max_t = self._n_online

        # Waiting pool for scheduled training events.
        # Format: {time_step: [(online_index, label), ...], ...}
        self._waiting_pool = {}

        # Schedule training events for each online sample.
        for i in range(self._n_online):
            commit_day = self._commit_days[i]
            y = self._online_Y[i]
            d = self._delays[i]
            label_val = self._get_label_value(y)
            if label_val == 0:
                event_time = commit_day + self.waiting_time
                self._schedule_event(i, event_time, 0)
            else:
                # If the delay is greater than the waiting time,
                # schedule a training event for the waiting time.
                if d > self.waiting_time:
                    event_time = commit_day + self.waiting_time
                    self._schedule_event(i, event_time, 0)
                event_time = commit_day + d
                self._schedule_event(i, event_time, 1)

    def _get_label_value(self, y) -> int:
        """
        Helper method to extract the label value from y.
        If y is one-hot encoded (i.e., a vector with more than one element),
        returns the index of the maximum value.
        Otherwise, returns the scalar value.
        """
        if self._is_tensor:
            if y.dim() > 0 and y.numel() > 1:
                return int(torch.argmax(y))
            else:
                return int(y.item())
        else:
            if y.ndim > 0 and y.size > 1:
                return int(np.argmax(y))
            else:
                return int(y)

    def _schedule_event(self, online_index: int, event_time: float, label: int) -> None:
        """
        Schedules a training event by finding the first online sample whose commit day is
        greater than or equal to event_time. If found, the event is assigned to that test time step.
        Otherwise, the event is ignored.
        """
        j = np.searchsorted(self._commit_days, event_time, side='left')
        if j >= self._n_online:
            # Do not schedule the event if event_time is beyond the last online sample.
            return
        event_time_step = j + 1  # Test time steps start at 1.
        if event_time_step not in self._waiting_pool:
            self._waiting_pool[event_time_step] = []
        self._waiting_pool[event_time_step].append((online_index, label))

    def fetch_initial_samples(self) -> Tuple[Union[np.ndarray, torch.Tensor],
                                              Union[np.ndarray, torch.Tensor]]:
        """
        Returns the pretraining samples with their true labels.
        """
        if self._n_pretrain <= 0:
            warnings.warn("No pretraining samples.", RuntimeWarning)
        self._next_is_pretrain = False
        return self._initial_X, self._initial_Y

    def next_test_and_train_samples(self) -> Tuple[
            Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None],
            Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Returns the test and training samples for the current time step.

        Test data: If the current time step is within 1 to n_online, returns the corresponding online sample
        (with its true label). Otherwise, returns None.

        Training data: If there are scheduled training events for the current time step, returns the corresponding
        training samples with their designated labels.
        """
        test_indices, test_xs, test_ys = self._next_test_samples()
        train_indices, train_xs, train_ys = self._next_train_samples()
        self._t += 1
        return test_indices, test_xs, test_ys, train_indices, train_xs, train_ys

    def _next_test_samples(self) -> Tuple[Union[np.ndarray, torch.Tensor, None],
                                           Union[np.ndarray, torch.Tensor, None],
                                           Union[np.ndarray, torch.Tensor, None]]:
        """
        Retrieves the test sample for the current time step.
        Returns the online sample index (starting from 0), its features, and its true label.
        """
        if self._t <= self._n_online:
            local_idx = self._t - 1
            xs = self._online_X[local_idx:local_idx + 1]  # Preserve original dimensions.
            ys = self._online_Y[local_idx:local_idx + 1]
            if self._is_tensor:
                global_idx = torch.tensor([local_idx], dtype=torch.long)
            else:
                global_idx = np.array([local_idx])
            return global_idx, xs, ys
        else:
            return None, None, None

    def _next_train_samples(self) -> Tuple[Union[np.ndarray, torch.Tensor, None],
                                            Union[np.ndarray, torch.Tensor, None],
                                            Union[np.ndarray, torch.Tensor, None]]:
        """
        Retrieves the training samples scheduled for the current time step (if any).
        """
        if self._t not in self._waiting_pool:
            return None, None, None
        indices_list = []
        xs_list = []
        ys_list = []
        for online_index, label in self._waiting_pool[self._t]:
            indices_list.append(online_index)
            xs_list.append(self._online_X[online_index:online_index + 1])
            ys_list.append(self._online_Y[online_index:online_index + 1])
        del self._waiting_pool[self._t]

        if self._is_tensor:
            indices_tensor = torch.tensor(indices_list, dtype=torch.long)
            xs_tensor = torch.cat(xs_list, dim=0)
            ys_tensor = torch.cat(ys_list, dim=0)
            return indices_tensor, xs_tensor, ys_tensor
        else:
            indices_array = np.array(indices_list)
            xs_array = np.concatenate(xs_list, axis=0)
            ys_array = np.concatenate(ys_list, axis=0)
            return indices_array, xs_array, ys_array

    def current_time_step(self) -> int:
        """
        Returns the current simulation time step (i.e., the sequential number of the test data).
        """
        return self._t

    def total_time_step(self) -> int:
        """
        Returns the total number of simulation time steps (based solely on the online test data count).
        """
        return self._max_t

    def has_next(self) -> bool:
        """
        Returns True if the current time step is within the online test data range.
        """
        return self._t <= self._max_t
