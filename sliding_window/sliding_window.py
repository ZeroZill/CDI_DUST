import numpy as np
from sklearn.neighbors import KDTree, BallTree
from sklearn.utils.validation import check_array

HIGH_DIM_THR = 30


class SlidingWindow:
    def __init__(self, window_size, n_classes):
        self.window_size = window_size
        self.n_classes = n_classes
        self.window = None  # shape: (n_samples, n_features)
        self.labels = None  # shape: (n_samples,), must be 1D, not one-hot

    def add_samples(self, samples, labels=None):
        samples = check_array(samples)
        n = samples.shape[0]

        if labels is None:
            labels = np.full((n,), -1, dtype=int)
        else:
            labels = np.asarray(labels)
            assert labels.ndim == 1, "labels must be one-dimensional"

        if self.window is None:
            self.window = samples
            self.labels = labels
        else:
            self.window = np.vstack((self.window, samples))
            self.labels = np.concatenate((self.labels, labels))

        if len(self.window) > self.window_size:
            excess = len(self.window) - self.window_size
            self.window = self.window[excess:]
            self.labels = self.labels[excess:]

    def _get_k_nearest(self, X, k, contains_unlabeled=True):
        if self.window is None or len(self.window) == 0:
            raise ValueError("Sliding window is empty.")

        if not contains_unlabeled:
            mask = self.labels != -1
            data = self.window[mask]
            label_pool = self.labels[mask]
        else:
            data = self.window
            label_pool = self.labels

        if len(data) == 0:
            raise ValueError("No labeled data in the sliding window.")

        k = min(k, len(data))

        if data.shape[1] >= HIGH_DIM_THR:
            tree = BallTree(data, metric="minkowski", p=1)
        else:
            tree = KDTree(data, metric="minkowski", p=2)

        dists, indices = tree.query(X, k=k)
        neighbors = label_pool[indices]
        return neighbors, dists

    def _vote_proba(self, neighbors, dists, mode):
        n_samples = neighbors.shape[0]
        proba = np.zeros((n_samples, self.n_classes))

        for i in range(n_samples):
            for j, cls in enumerate(neighbors[i]):
                if cls == -1:
                    continue
                if mode == "majority":
                    vote_value = 1
                elif mode == "gravity":
                    vote_value = 1 / (dists[i, j] ** 2 + 1e-6)
                else:
                    raise ValueError(f"Unsupported voting mode: {mode}")
                proba[i, cls] += vote_value

        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict_proba(self, X, k=3, mode="majority", return_neighbors=False):
        X = check_array(X)
        ks = [k] if isinstance(k, int) else sorted(np.asarray(k).astype(int))

        tot_proba = np.zeros((X.shape[0], self.n_classes))
        final_neighbors = None

        for ki in ks:
            neighbors, dists = self._get_k_nearest(X, ki, contains_unlabeled=False)
            proba_k = self._vote_proba(neighbors, dists, mode)
            tot_proba += proba_k

            if ki == ks[-1] and return_neighbors:
                final_neighbors = neighbors

        tot_proba /= np.sum(tot_proba, axis=1, keepdims=True)

        return (tot_proba, final_neighbors) if return_neighbors else tot_proba

    def predict(self, X, k=3, mode="majority", return_neighbors=False):
        result = self.predict_proba(X, k, mode, return_neighbors)
        if return_neighbors:
            proba, neighbors = result
            return np.argmax(proba, axis=1), neighbors
        else:
            proba = result
            return np.argmax(proba, axis=1)
