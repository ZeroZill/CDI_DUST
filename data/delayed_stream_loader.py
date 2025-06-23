import numbers
import warnings

import numpy as np
from typing import List, Union, Tuple


class DelayedStreamLoader:
    r"""
    Initializes the DelayedStreamLoader object.

    Parameters
    ----------
    X: List or np.ndarray
        Feature vectors.
    Y: List or np.ndarray
        Array of labels. The lengths of X and Y must be the same.
    time_steps: List, np.ndarray or None, optional (default=None)
        Time steps (at least 1). Default is None. If time_steps is None,
        it defaults to [1, 2, ..., n], where n is the length of Y.
    delays: float, List, np.ndarray or None, optional (default=None)
        Delay values. Default is None. If delays is None,
        it defaults to zero delays for all samples. And If delays is a
        single value, it is applied to all samples.
    n_pretrain: int or None, optional (default=None)
        Number of initial samples for pretraining. Default is None. If
        n_pretrain is None, it defaults to no initial sample.
    """

    def __init__(self,
                 X: Union[List, np.ndarray], Y: Union[List, np.ndarray],
                 time_steps: Union[None, List[int], np.ndarray] = None,
                 delays: Union[None, float, List[float], np.ndarray] = None,
                 n_pretrain: Union[None, int] = None):
        assert len(X) == len(Y), "The number of features and labels must be the same"
        self._next_is_pretrain = False
        self._n_pretrain = 0
        self._initial_X, self._initial_Y = None, None
        if n_pretrain is not None and n_pretrain > 0:
            self._next_is_pretrain = True
            self._n_pretrain = min(n_pretrain, len(Y))
            self._initial_X, self._initial_Y = X[:self._n_pretrain], Y[:self._n_pretrain]
        self._n_online_samples = len(Y) - self._n_pretrain
        time_steps = self._validate_time_steps(time_steps)
        delays = self._validate_delay_values(delays)
        # X = np.reshape(X, (self.n_samples, -1))
        # Y = np.reshape(Y, (self.n_samples, -1))
        self._stream = {
            "X": X[self._n_pretrain:],
            "Y": Y[self._n_pretrain:],
            "time_steps": time_steps,
            "delays": delays
        }
        self._t = 1
        self._waiting_pool = {}

    def set_delays(self, delays: Union[float, List[float], np.ndarray]) -> None:
        """
        Sets the delay values.

        Parameters
        ----------
        delays: List or np.ndarray
            New Delay values. Cannot be None.
        """
        delays = self._validate_delay_values(delays)
        self._stream["delays"] = delays

    def _validate_time_steps(self, time_steps: Union[None, List[int], np.ndarray]) -> np.ndarray:
        if time_steps is None:
            time_steps = np.arange(1, self._n_online_samples + 1)
        else:
            assert len(time_steps) == self._n_online_samples, \
                "The number of time steps must be the same as the number of online samples"
            assert np.min(time_steps) >= 1
            time_steps = np.asarray(time_steps)
        self._max_t = np.max(time_steps)
        return time_steps

    def _validate_delay_values(self, delays: Union[None, float, List[float]]) -> np.ndarray:
        if delays is None:
            delays = np.zeros(self._n_online_samples)
        elif isinstance(delays, numbers.Number):
            delays = np.ones(self._n_online_samples) * delays
        else:
            assert len(delays) == self._n_online_samples, \
                "The number of delay values must be the same as the number of online samples"
            delays = np.asarray(delays)
        return delays

    def fetch_initial_samples(self) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None]]:
        if self._n_pretrain <= 0:
            warnings.warn("There is no initial samples.", RuntimeWarning)
        self._next_is_pretrain = False
        return self._initial_X, self._initial_Y

    def next_test_and_train_samples(self) -> Tuple[
        Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None],
        Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Retrieves the next test and train samples.

        Returns
        -------
        Tuple containing the test feature vectors, test labels, train feature vectors, and train labels.
        If there are no train samples available, None is returned for the train feature vectors and labels.
        """
        if self._next_is_pretrain:
            warnings.warn("Initial samples are not fetched, please fetch the initial samples first.",
                          RuntimeWarning)
        test_indices, test_xs, test_ys = self._next_test_samples()
        train_indices, train_xs, train_ys = self._next_train_samples()
        self._t += 1
        return test_indices, test_xs, test_ys, train_indices, train_xs, train_ys

    def current_time_step(self) -> int:
        return self._t

    def total_time_step(self) -> int:
        return self._n_online_samples

    def has_next(self) -> bool:
        return self._t <= self._max_t

    def _next_test_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves the next test samples and add unlabeled samples into waiting pool.

        Returns
        -------
        Tuple containing the test feature vectors and test labels.
        """
        test_sample_indices = np.where(self._stream["time_steps"] == self._t)[0]
        xs, ys, time_steps, delays = self._stream["X"][test_sample_indices], \
                                     self._stream["Y"][test_sample_indices], \
                                     self._stream["time_steps"][test_sample_indices], \
                                     self._stream["delays"][test_sample_indices]

        # Add unlabeled samples into waiting pool
        for index, x, y, time_step, delay in zip(test_sample_indices, xs, ys, time_steps, delays):
            if np.isinf(delay):
                label_available_step = float("inf")
            else:
                label_available_step = int(np.ceil(time_step + delay))
            if label_available_step not in self._waiting_pool:
                self._waiting_pool[label_available_step] = []
            self._waiting_pool[label_available_step].append(index)
        return test_sample_indices, xs, ys

    def _next_train_samples(self) -> Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        """
        Retrieves the next train samples.

        Returns
        -------
        Tuple containing the train feature vectors and train labels.
        If no train samples are available, None is returned.
        """
        if self._t not in self._waiting_pool:
            indices, xs, ys = None, None, None
        else:
            label_available_indices = self._waiting_pool[self._t]
            xs, ys, time_steps, delays = self._stream["X"][label_available_indices], \
                                         self._stream["Y"][label_available_indices], \
                                         self._stream["time_steps"][label_available_indices], \
                                         self._stream["delays"][label_available_indices]
            del self._waiting_pool[self._t]
            indices = np.asarray(label_available_indices)
        return indices, xs, ys
