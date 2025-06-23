import time
from typing import Union

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.io import arff
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from skmultiflow.data import SEAGenerator, HyperplaneGenerator, LEDGeneratorDrift, AGRAWALGenerator, LEDGenerator, \
    RandomRBFGeneratorDrift, SineGenerator, ConceptDriftStream

from config import EXP_DATASETS


def randsphere_uniform(N: int, d: int, random_state: Union[int, None, np.random.RandomState] = None) -> np.ndarray:
    """
    Returns an N by d array, X, in which each of the m rows has the n Cartesian coordinates of a random point
    uniformly-distributed over the interior of a d-dimensional hypersphere with radius 1 and center at the origin.

    Parameters:
        N: int
            The number of random points to generate.

        d: int
            The number of dimensions for each point.

        random_state: int, numpy.random.RandomState or None, default=None
                Random generator for sampling.

    Returns:
        numpy.ndarray: An m by n array containing the Cartesian coordinates of the random points within the hypersphere.
    """
    random_state = check_random_state(random_state)
    # Randomly generate and normalize direction vectors
    random_directions = random_state.randn(N, d)
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
    # Generate random distances from the center
    random_dists = random_state.rand(N, 1) ** (1 / d)
    # Scale the direction vectors to obtain the final random points
    X = random_directions * random_dists
    return X


def generate_delays(delay_mode, delay, n_samples, label_ratio=1.0, random_state=0, return_critical_value=False):
    """
    Generate delay array based on delay mode (fixed/varying) and label availability ratio.

    Parameters:
        delay_mode: str
            Either "fixed" or "varying".

        delay: float
            For "fixed": constant delay value.
            For "varying": mean of gamma distribution.

        n_samples: int
            Number of delays to generate.

        label_ratio: float, default=1.0
            Proportion of available labels (must be between 0 and 1 inclusive).

        random_state: int, np.random.RandomState, or None
            Random seed or generator.

        return_critical_value: bool
            Whether to return 95% gamma critical value when using varying delays.

    Returns:
        delays: np.ndarray
            Array of delays with shape (n_samples,).
        critical (optional): float
            The 95% gamma critical value (only when delay_mode == "varying" and return_critical_value=True).
    """
    if not (0 <= label_ratio <= 1):
        raise ValueError("label_ratio must be between 0 and 1.")

    random_state = check_random_state(random_state)
    n_available = int(n_samples * label_ratio)
    n_unavailable = n_samples - n_available
    unavailable_delays = np.full(n_unavailable, np.inf)

    if delay_mode == "fixed":
        available_delays = np.full(n_available, delay)
        delays = np.concatenate([available_delays, unavailable_delays])
        if 0.0 < label_ratio < 1.0:
            random_state.shuffle(delays)
        return delays

    elif delay_mode == "varying":
        shape = 2
        scale = delay / shape
        available_delays = random_state.gamma(shape, scale, n_available)
        delays = np.concatenate([available_delays, unavailable_delays])
        if 0.0 < label_ratio < 1.0:
            random_state.shuffle(delays)
        if return_critical_value:
            critical = stats.gamma.ppf(0.95, shape, scale=scale)
            return delays, critical
        else:
            return delays

    else:
        raise ValueError("Invalid delay_mode. Must be 'fixed' or 'varying'.")


def normalization(X):
    print("[Normalization]\tStart!")
    is_tensor = isinstance(X, torch.Tensor)
    start = time.time()
    scaler = StandardScaler()  # When clustering is involved, standard scaler is better
    scaler.fit(X)
    X = scaler.transform(X)
    print("[Normalization]\tOver! Time consumed: {} (s)".format(time.time() - start))
    return torch.tensor(X, dtype=torch.float32) if is_tensor else X


def load_dataset_from_arff(dataset, path):
    n_features, _, n_samples = EXP_DATASETS[dataset]
    data, meta = arff.loadarff(path)
    data = pd.DataFrame(data)
    for column in data.columns:
        data_type, attrs = meta[column]
        if data_type != "numeric":
            attr_mapping = {attr.encode(): idx for idx, attr in enumerate(attrs)}
            data[column] = data[column].map(attr_mapping)
    data = data.to_numpy()
    X, Y = data[:, :n_features], data[:, n_features].astype(int)
    if n_samples is not None:
        X, Y = X[:n_samples], Y[:n_samples]
    return X, Y


def load_dataset_from_csv(dataset, path, header=None):
    n_features, _, n_samples = EXP_DATASETS[dataset]
    data = pd.read_csv(path, header=header, engine='python', on_bad_lines='skip')
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values

    unique_labels = np.unique(Y)
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    Y = np.array([label_mapping[label] for label in Y])

    if n_samples is not None:
        X, Y = X[:n_samples], Y[:n_samples]

    return X, Y


def load_jit_sdp_dataset(dataset, first_n=None):
    """
    The CSV file has the following structure (with a header row):
    - Column 0: commit unix timestamp (sorted by time)
    - Columns 1 to 14: features (14 columns, as float)
    - Column 15: label (0 or 1, as int)
    - Last column: delay (in days, as float) representing the delay of the label availability.
    """
    dataset_paths = {
        "JIT-SDP_django": "datasets/django_vld_st.csv",
        "JIT-SDP_pandas": "datasets/pandas_vld_st.csv",
    }

    if dataset not in dataset_paths:
        raise ValueError(f"Dataset {dataset} is not recognized. Available datasets: {list(dataset_paths.keys())}")

    # Read CSV file with no header row (first row is the header in the file)
    data = pd.read_csv(dataset_paths[dataset], header=0, engine='python', on_bad_lines='skip')

    if first_n is not None:
        data = data.iloc[:first_n, :]

    # Extract columns:
    # Column 0: commit unix timestamp (ensure int64)
    commit_unix_times = data.iloc[:, 0].astype(np.int64).to_numpy()

    # Columns 1 to 14 (inclusive): features X (as float)
    X = data.iloc[:, 1:15].astype(float).to_numpy()

    # Column 15: label Y (as int)
    Y = data.iloc[:, 15].astype(int).to_numpy()

    # Last column: delay (as float)
    delay = data.iloc[:, -1].astype(float).to_numpy()

    return X, Y, commit_unix_times, delay


def load_dataset(dataset, datasets_meta=EXP_DATASETS, first_n=None):
    random_state = check_random_state(0)
    n_features, n_classes, n_samples = datasets_meta[dataset]
    if "Inc_Hyperplane" in dataset:
        mag_changes = {'1': 0.001, '2': 0.01, '3': 0.1}
        idx = dataset[-1]
        data_gen = HyperplaneGenerator(random_state=random_state, mag_change=mag_changes[idx],
                                       n_drift_features=5, sigma_percentage=0.0)
        X, Y = data_gen.next_sample(n_samples)
    elif "Inc_RBF" in dataset:
        change_speeds = {'1': 0.0001, '2': 0.001, '3': 0.01}
        idx = dataset[-1]
        data_gen = RandomRBFGeneratorDrift(model_random_state=random_state,
                                           sample_random_state=random_state,
                                           change_speed=change_speeds[idx],
                                           num_drift_centroids=3,
                                           n_centroids=20)
        X, Y = data_gen.next_sample(n_samples)
    elif "SEA" in dataset and len(dataset.split('_')) == 2:
        cmd = dataset.split('_')[1]
        configs = {
            "r": [1, 3, 1, 3],
            "n": [2, 0, 1, 3],
        }
        config = configs[cmd[:1]]
        is_gradual = len(cmd) > 1 and cmd[1] == 'g'
        n_samples_per_chunk = n_samples // len(config)
        if not is_gradual:
            X, Y = [], []
            for function in config:
                data_gen = SEAGenerator(random_state=random_state, classification_function=function,
                                        balance_classes=True)
                next_X, next_Y = data_gen.next_sample(n_samples_per_chunk)
                X.append(next_X)
                Y.append(next_Y)
            X = np.vstack(X)
            Y = np.concatenate(Y)
        else:
            gradual_width = 1000
            data_gen = SEAGenerator(random_state=random_state, classification_function=config[0],
                                    balance_classes=True)
            for i in range(1, len(config)):
                data_gen = ConceptDriftStream(data_gen,
                                              SEAGenerator(random_state=random_state,
                                                           classification_function=config[i],
                                                           balance_classes=True),
                                              position=n_samples_per_chunk * i,
                                              width=gradual_width,
                                              random_state=random_state)
            X, Y = data_gen.next_sample(n_samples)
            Y = Y.astype(int)
    elif "Agrawal" in dataset and len(dataset.split('_')) == 2:
        cmd = dataset.split('_')[1]
        configs = {
            "r": [2, 4, 2, 4],
            "n": [3, 2, 5, 7],
        }
        config = configs[cmd[:1]]
        is_gradual = len(cmd) > 1 and cmd[1] == 'g'
        n_samples_per_chunk = n_samples // len(config)
        if not is_gradual:
            X, Y = [], []
            for function in config:
                data_gen = AGRAWALGenerator(random_state=random_state, classification_function=function,
                                            balance_classes=True)
                next_X, next_Y = data_gen.next_sample(n_samples_per_chunk)
                X.append(next_X)
                Y.append(next_Y)
            X = np.vstack(X)
            Y = np.concatenate(Y)
        else:
            gradual_width = 1000
            data_gen = AGRAWALGenerator(random_state=random_state, classification_function=config[0],
                                        balance_classes=True)
            for i in range(1, len(config)):
                data_gen = ConceptDriftStream(data_gen,
                                              AGRAWALGenerator(random_state=random_state,
                                                               classification_function=config[i],
                                                               balance_classes=True),
                                              position=n_samples_per_chunk * i,
                                              width=gradual_width,
                                              random_state=random_state)
            X, Y = data_gen.next_sample(n_samples)
            Y = Y.astype(int)
    elif "LED" in dataset and len(dataset.split('_')) == 2:
        cmd = dataset.split('_')[1]
        configs = {
            "r": [0, 1, 0, 1],
            "n": [0, 1]
        }
        config = configs[cmd[:1]]
        is_gradual = len(cmd) > 1 and cmd[1] == 'g'
        n_samples_per_chunk = n_samples // len(config)
        if not is_gradual:
            X, Y = [], []
            for function in config:
                if function == 0:
                    data_gen = LEDGenerator(random_state=random_state, has_noise=True,
                                            noise_percentage=0.1)
                else:
                    data_gen = LEDGeneratorDrift(random_state=random_state, has_noise=True,
                                                 n_drift_features=3,
                                                 noise_percentage=0.1)

                next_X, next_Y = data_gen.next_sample(n_samples_per_chunk)
                X.append(next_X)
                Y.append(next_Y)
            X = np.vstack(X)
            Y = np.concatenate(Y)
        else:
            gradual_width = 1000
            if config[0] == 0:
                data_gen = LEDGenerator(random_state=random_state, has_noise=True,
                                        noise_percentage=0.1)
            else:
                data_gen = LEDGeneratorDrift(random_state=random_state, has_noise=True,
                                             n_drift_features=3,
                                             noise_percentage=0.1)
            for i in range(1, len(config)):
                if config[i] == 0:
                    next_data_gen = LEDGenerator(random_state=random_state, has_noise=True,
                                                 noise_percentage=0.1)
                else:
                    next_data_gen = LEDGeneratorDrift(random_state=random_state, has_noise=True,
                                                      n_drift_features=3,
                                                      noise_percentage=0.1)
                data_gen = ConceptDriftStream(data_gen,
                                              next_data_gen,
                                              position=n_samples_per_chunk * i,
                                              width=gradual_width,
                                              random_state=random_state)
                X, Y = data_gen.next_sample(n_samples)
                Y = Y.astype(int)
    elif "Sine" in dataset and len(dataset.split('_')) == 2:
        cmd = dataset.split('_')[1]
        configs = {
            "r": [0, 2, 0, 2],
            "n": [0, 2]
        }
        config = configs[cmd[:1]]
        is_gradual = len(cmd) > 1 and cmd[1] == 'g'
        n_samples_per_chunk = n_samples // len(config)
        if not is_gradual:
            X, Y = [], []
            for function in config:
                data_gen = SineGenerator(classification_function=function, balance_classes=True,
                                         random_state=random_state)
                next_X, next_Y = data_gen.next_sample(n_samples_per_chunk)
                X.append(next_X)
                Y.append(next_Y)
            X = np.vstack(X)
            Y = np.concatenate(Y)
        else:
            gradual_width = 1000
            data_gen = SineGenerator(classification_function=config[0], balance_classes=True,
                                     random_state=random_state)
            for i in range(1, len(config)):
                data_gen = ConceptDriftStream(data_gen,
                                              SineGenerator(classification_function=config[i], balance_classes=True,
                                                            random_state=random_state),
                                              position=n_samples_per_chunk,
                                              width=gradual_width,
                                              random_state=random_state)
            X, Y = data_gen.next_sample(n_samples)
            Y = Y.astype(int)
    elif dataset == "Electricity":
        data = sio.loadmat("datasets/electricitypricing.mat")["data"]
        X, Y = data[:, :8], data[:, 9].astype(int)
    elif dataset == "Weather":
        data = sio.loadmat("datasets/weather.mat")["data"]
        X, Y = data[:, :8], data[:, 9].astype(int)
    elif dataset == "Airlines":
        X, Y = load_dataset_from_arff("Airlines", "datasets/airlines.arff")
    elif dataset == "Rialto":
        X, Y = load_dataset_from_arff("Rialto", "datasets/rialto.arff")
    elif dataset == "INS-Inc-Reo":
        X, Y = load_dataset_from_arff("INS-Inc-Reo", "datasets/INSECTS-incremental-reoccurring_balanced_norm.arff")
    elif dataset == "INS-Inc-Abt":
        X, Y = load_dataset_from_arff("INS-Inc-Abt", "datasets/INSECTS-incremental-abrupt_balanced_norm.arff")
    elif dataset == "INS-Grad":
        X, Y = load_dataset_from_arff("INS-Grad", "datasets/INSECTS-gradual_balanced_norm.arff")
    elif dataset == "Covtype":
        X, Y = load_dataset_from_arff("Covtype", "datasets/covtype.arff")
    elif dataset == "Poker_hand":
        X, Y = load_dataset_from_arff("Poker_hand", "datasets/poker.arff")
    elif dataset == "Asfault":
        X, Y = load_dataset_from_csv("Asfault", "datasets/Asfault.csv")
        X = PCA(n_components=500).fit_transform(X)
    elif dataset == "UWave":
        X, Y = load_dataset_from_csv("UWave", "datasets/UWave.csv")
    else:
        raise ValueError("Unknown dataset.")
    if first_n is not None:
        return X[:first_n], Y[:first_n]
    return X, Y
