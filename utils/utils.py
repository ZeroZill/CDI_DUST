import os
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


def check_array(X, ensure_2d=True) -> Union[np.ndarray, torch.Tensor]:
    """
    Function to check and convert input data to a numpy array.

    Parameters:
        X: array-like or pandas DataFrame
            The input data to be checked.

        ensure_2d: bool, optional (default=True)
            If True, ensures that the resulting array is 2-dimensional.
            If False, the resulting array can be 1-dimensional if possible.

    Returns:
        numpy.ndarray or torch.Tensor:
            The converted numpy array from the input data.
    """
    if isinstance(X, np.ndarray):
        if ensure_2d:
            return X.reshape(1, -1) if X.ndim == 1 else X
        else:
            return X
    elif isinstance(X, torch.Tensor):
        if ensure_2d:
            return X.unsqueeze(0) if X.ndim == 1 else X
        else:
            return X

    try:
        # Try converting X to numpy array
        X_array = np.asarray(X)
        if ensure_2d:
            return X_array.reshape(1, -1) if X_array.ndim == 1 else X_array
        else:
            return X_array
    except Exception as e:
        raise ValueError("Input data could not be converted to a numpy array: {}".format(e))


def check_data(X, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Checks if the input data X and y are valid and compatible.

    Parameters:
    X : array-like or pd.DataFrame
        Input feature matrix.

    y : array-like or pd.Series
        Input target labels.

    Raises:
    ValueError:
        If X and y have inconsistent shapes or if they are not of compatible types.

    Returns:
    numpy.ndarray, numpy.ndarray
        Validated feature matrix X and target labels y as numpy arrays.
    """
    X = check_array(X)

    if isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    else:
        y = np.asarray(y)

    if len(X) != len(y):
        raise ValueError("Input X and y have inconsistent lengths.")

    return X, y


def check_weights(weights, n_samples) -> np.ndarray:
    """
    Check and validate the sample weights.
    - If weights is None, it will be set to 1 (equal weights for all samples).
    - If weights is an integer or a float, it will be broadcast to a 1D array with the same value for all samples.
    - If weights is array-like, it will be converted to a 1D numpy array and checked for the correct length (n_samples).

    Parameters:
        weights : int, float, array-like, or None
            The input sample weights.

        n_samples : int
            The number of samples in the dataset.

    Returns:
        numpy.ndarray
            Validated and converted sample weights as a 1D numpy array.
    """
    if weights is None:
        weights = 1

    # If weights is an integer or a float, broadcast it to a 1D array with the same value for all samples
    if isinstance(weights, (int, float)):
        weights = np.ones(n_samples) * weights

    else:
        # If weights is array-like, convert it to a 1D numpy array
        weights = np.asarray(weights)

        # Check if the number of sample weights matches the number of samples
        if len(weights) != n_samples:
            raise ValueError("The number of sample weights must match the number of samples.")

    return weights


def is_confident(proba1, proba2, proba_threshold):
    pred1 = np.argmax(proba1)
    pred2 = np.argmax(proba2)

    if pred1 != pred2:
        return False

    max_proba1 = proba1[pred1]
    max_proba2 = proba2[pred2]

    if max_proba1 < proba_threshold or max_proba2 < proba_threshold:
        return False

    return True


def gmean(y_true, y_pred):
    y_true_count = {cls: 0 for cls in set(y_true)}
    y_pred_count = {cls: 0 for cls in set(y_pred)}
    for yt, yp in zip(y_true, y_pred):
        y_true_count[yt] += 1
        if yt == yp:
            y_pred_count[yp] += 1

    recalls = np.prod([(y_pred_count[cls] / y_true_count[cls]) for cls in list(y_pred_count.keys())])
    res = np.power(recalls, 1 / len(y_pred_count))
    return res


def matrix_2_norm(A):
    try:
        A = check_array(A)
        eigenvalues, _ = torch.linalg.eig(A @ A.T)
        eigenvalues = eigenvalues.real
        norm_2 = torch.sqrt(torch.max(eigenvalues))
        return norm_2
    except RuntimeError:
        return torch.tensor(torch.nan)


def check_dir(dir_path):
    if dir_path != '' and not os.path.exists(dir_path):
        os.makedirs(dir_path)
        return False
    return True


def check_path(path, delete_if_exists=False):
    parent_dir = os.path.dirname(path)

    path_exist = os.path.exists(path)

    if delete_if_exists and path_exist:
        os.remove(path)

    if parent_dir != '' and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    return path_exist


def check_single_result_exists(path, dataset, model, delay, delay_mode, label_ratio, seed, return_res=False):
    df = pd.read_csv(path)

    row = df[
        (df['dataset'] == dataset) &
        (df['model'] == model) &
        (df['delay'] == delay) &
        (df['label_ratio'] == label_ratio) &
        (df['delay_mode'] == delay_mode) &
        (df['seed'] == seed)
        ]

    row_exists = not row.empty

    if return_res:
        if row_exists:
            acc = row['acc'].values[0]
            time = row['time'].values[0]
            return True, (acc, time)
        else:
            return False, (None, None)
    else:
        return row_exists


def check_single_tuning_result_exists(path, dataset, model, delay, delay_mode, params, seed, return_res=True):
    df = pd.read_csv(path)

    mask = (
            (df['dataset'] == dataset) &
            (df['model'] == model) &
            (df['delay'] == delay) &
            (df['delay_mode'] == delay_mode) &
            (df['seed'] == seed)
    )
    for key, val in params.items():
        mask &= (df[key] == val)

    subset = df[mask]
    exists = not subset.empty

    if not return_res:
        return exists

    if exists:
        final_acc = subset['final_acc'].iloc[0]
        time_cost = subset['time_cost'].iloc[0]
        return True, (final_acc, time_cost)
    else:
        return False, (None, None)


def save_result(result_path, records):
    check_path(result_path, delete_if_exists=True)
    data = pd.DataFrame(records)
    data.to_csv(result_path, index=False)


def to_tensor(X, Y, n_classes):
    # To tensor
    X_tensor, Y_tensor = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

    # Change labels to one-hot encoding
    Y_tensor = F.one_hot(Y_tensor, n_classes)

    return X_tensor, Y_tensor
