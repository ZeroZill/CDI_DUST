import itertools
import os
import pickle
import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree, BallTree
from sklearn.utils import check_random_state

from micro_cluster import MicroCluster
from utils import check_array, check_data, check_weights, check_dir
from stability_indicators.cdi import calc_cdi_based_on_MCS

HIGH_DIM_THR = 30


class MicroClusterSystem:
    def __init__(self, classes, max_n_MCs, decay_factor, weight_threshold,
                 involve_range=1.0,
                 conflict_mode="dominate",  # "deduct" or "dominate"
                 weight_penalty=0.5,
                 weight_confidence_level=0.95,
                 min_n_MCs_per_class=10,
                 maintain_cov=True,
                 random_state=None,
                 auto_tune=True):
        """
        MicroClusterSystem maintains all MicroClusters of all classes.

        Parameters:
            classes: list
                A list of class labels for which micro clusters will be maintained.

            max_n_MCs: int
                The maximum number of micro clusters that will be maintained for each class.

            decay_factor: float
                The decay factor used for decaying the weight of micro clusters over time.

            weight_threshold: float
                The weight threshold below which micro clusters will be removed.

            involve_range: float
                A number determines whether to involve a near-by sample into current micro cluster. If the distance
                between the sample and the center of the micro cluster is not larger than involve range, it will be
                involved.

            conflict_mode: str, "deduct" or "dominate"
                The mode for determining whether conflict happens.

            weight_penalty: float
                The weight deduction for one micro cluster when it is found to be conflicted with other micro clusters

            min_n_MCs_per_class: int
                The minimum number of micro clusters one class should maintain, even they should be expired

            random_state: int, numpy.random.RandomState or None, optional (default=None)
                Random generator for sampling.
        """
        self.n_features = None
        self.max_n_MCs = max_n_MCs
        self.decay_factor = decay_factor
        self.classes = classes

        self.maintain_cov = maintain_cov

        self.weight_threshold = weight_threshold
        self.weight_penalty = weight_penalty

        # self.slack_factor = slack_factor
        self._involve_range = involve_range

        self._to_auto_tune = auto_tune

        if conflict_mode not in ["deduct", "dominate"]:
            raise ValueError(f"Invalid conflict mode: {conflict_mode}")

        self.conflict_mode = conflict_mode
        if self.conflict_mode == "dominate":
            assert weight_confidence_level is not None, \
                f"The conflict mode 'dominate' requires weight confidence level to be specified."
            self.weight_confidence_level = weight_confidence_level

        self.min_n_MCs_per_cls = min_n_MCs_per_class

        self._MCs_repo = {cls: [] for cls in [*self.classes, -1]}  # class value -1 stands for unlabeled data
        self._MCs_indices = {}

        self._random_state = check_random_state(random_state)

        self.n_samples_of_cls = np.zeros(max(self.classes) + 1)

        self._initialized = False
        self.t = 0

    @property
    def n_MCs(self):
        return len(self._MCs_indices)

    def reset(self):
        self._MCs_repo = {cls: [] for cls in [*self.classes, -1]}  # class value -1 stands for unlabeled data
        self._MCs_indices = {}

        self._initialized = False
        self.t = 0

    def set_seed(self, seed):
        self._random_state = check_random_state(seed)

    def set_time(self, t):
        self.t = t

    def initialize_MCs(self, X, y, k, sample_weights=None):
        """
        Initializes micro clusters for each class in the dataset using an offline clustering method.

        Parameters:
            X: numpy.ndarray, shape (n_samples, n_features)
                The input data samples.

            y: numpy.ndarray, shape (n_samples,)
                The class labels corresponding to each data sample.

            k: int
                The default number of micro clusters of each class.

            sample_weights: None, int, float or array-like, shape (n_samples,), optional (default=1)
                The weight of each sample. If an integer is provided, the same weight is used for all samples.

        Returns:
            None
        """
        X, y = check_data(X, y)

        self.n_features = X.shape[1]

        if self._to_auto_tune:
            self._involve_range = 0.35 * self._calc_involve_range_basis(X, y, mode="std")
        unique_labels, counts = np.unique(y, return_counts=True)

        for label, count in zip(unique_labels, counts):
            self.n_samples_of_cls[label] += count

        n_samples, _ = X.shape
        sample_weights = check_weights(sample_weights, n_samples)

        separated_samples = {label: (X[y == label], sample_weights[y == label]) for label in unique_labels}

        for label, (samples, weights) in separated_samples.items():
            n_samples = len(samples)

            if n_samples <= 0:
                continue

            n_clusters = min(k, n_samples)

            # Here, k-means is applied as the offline clustering method
            offline_clustering_method = KMeans(n_clusters=n_clusters, random_state=self._random_state)

            # Perform offline clustering using the specified method
            offline_clustering_method.fit(samples)
            indices = offline_clustering_method.labels_
            unique_indices = np.unique(indices)

            single_point_MCs = []
            min_nonzero_r = self._involve_range

            for idx in unique_indices:
                mc = MicroCluster(cur_t=self.t, cls=label, classes=self.classes, maintain_cov=self.maintain_cov)

                # Gather all samples belong to current cluster
                samples_of_cur_cluster = samples[indices == idx]
                sample_weights_of_cur_cluster = weights[indices == idx]
                # Insert samples into the micro cluster
                mc.insert_sample(cur_t=self.t, samples=samples_of_cur_cluster,
                                 sample_weights=sample_weights_of_cur_cluster)
                if mc.n_samples == 1:
                    single_point_MCs.append(mc)
                else:
                    min_nonzero_r = min(min_nonzero_r, mc.radius)
                mc.in_range(label, n_samples=len(samples_of_cur_cluster))
                self._append_MC(mc)

            for single_point_MC in single_point_MCs:
                single_point_MC.radius = min_nonzero_r

        # Initialization over
        self._initialized = True

    def partial_fit(self, X, y, sample_weights=None, is_pseudo=False):
        """
        Incrementally update the MicroClusterSystem using new data samples.

        Parameters:
            X: numpy.ndarray, shape (n_samples, n_features)
                The input data samples.

            y: numpy.ndarray, shape (n_samples,), optional (default=-1)
                The class labels corresponding to each data sample.

            sample_weights: None, int, float or array-like, shape (n_samples,), optional (default=1)
                The weight of each sample. If an integer is provided, the same weight is used for all samples.

            is_pseudo: bool
                Whether this partial_fit is for pseudo-labeled data or delayed labeled data

        Returns:
            list or None: List of micro-clusters that were emphasized due to class changes, or None if no
            emphasis occurred.
        """
        if not self._initialized:
            raise RuntimeError(
                "The `fit()` method of MicroClusterSystem can be invoked only after it has been initialized.")

        X, y = check_data(X, y)
        n_samples, n_dim = X.shape
        sample_weights = check_weights(sample_weights, n_samples)

        conflict = False
        emphasized_MCs = []
        removed_MCs = []

        for single_x, single_y, single_weight in zip(X, y, sample_weights):
            self.n_samples_of_cls[single_y] += 1
            # Seems I have to rebuild the kd-tree every time
            nearest_MC, nearest_dist = self._get_k_nearest_MCs(single_x, k=1, contains_unlabeled=True)
            nearest_MC, nearest_dist = nearest_MC[0, 0], nearest_dist[0, 0]

            # Out of range
            if nearest_dist > nearest_MC.radius:
                new_MC = self._create_new_MC_on_point(self.t, single_x, single_y, nearest_MC.radius, single_weight)

            # labels unmatched
            elif single_y != -1 and nearest_MC.cls != -1 and single_y != nearest_MC.cls:
                nearest_MC.in_range(single_y)
                deleted = False
                if self.conflict_mode == "deduct":
                    if self.weight_penalty > 0:
                        nearest_MC.weight -= self.weight_penalty
                    else:
                        nearest_MC.weight -= 1 / nearest_MC.n_samples
                    # If old micro cluster is gone, means the new one needs to be enhanced
                    if nearest_MC.weight <= self.weight_threshold:
                        self._delete_MC(nearest_MC)
                        removed_MCs.append(nearest_MC)
                        deleted = True
                    # if new_MC.debug_mode:
                    #     new_MC.nearest_MC = nearest_MC
                elif self.conflict_mode == "dominate":
                    if nearest_MC.weight_row[nearest_MC.cls] / np.sum(nearest_MC.weight_row) \
                            < self.weight_confidence_level:
                        self._delete_MC(nearest_MC)
                        removed_MCs.append(nearest_MC)
                        deleted = True
                new_MC = \
                    self._create_new_MC_on_point(self.t, single_x, single_y, nearest_MC.radius, single_weight)

                # If deletion is executed, emphasize the new MC.
                # DO NOT CHANGE THE STATEMENTS ORDER HERE
                if deleted:
                    conflict = True
                    emphasized_MCs.append(new_MC)
            # labels matched
            else:
                # If nearest MC is not labeled, label it and move it to corresponding repo
                if is_pseudo:
                    if nearest_MC.cls == -1 and nearest_MC.cls != single_y:
                        self._labeling_unlabeled_MC(nearest_MC, single_y)
                    nearest_MC.insert_sample(self.t, single_x, single_y, single_weight)
                    nearest_MC.in_range(single_y)

        return (False, None, None) if not conflict else (True, emphasized_MCs, removed_MCs)

    def overall_sampling(self, sampling_num_for_one_MC=-1):
        """
        Perform overall sampling across all micro clusters.

        Args:
            sampling_num_for_one_MC: int
                Number of samples to be drawn from each micro cluster.

        Returns:
            np.ndarray, np.ndarray: Stacked samples and concatenated labels from all micro clusters.
        """
        # Initialize empty lists for samples and labels
        samples = []
        labels = []

        # Iterate through all micro clusters
        for mc in self.all_MCs(contains_unlabeled=False):
            # Perform sampling in the current micro cluster
            new_samples, new_labels = self.sampling_from_MC(mc, sampling_num_for_one_MC)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(new_labels)

        # Concatenate samples and labels from all micro clusters
        samples = np.vstack(samples)
        labels = np.concatenate(labels)

        return samples, labels

    def sampling_by_KNN(self, MCs, k, extra_sampling_num=None):
        # Initialize empty lists for samples and labels
        samples = []
        labels = []

        # Iterate through all micro clusters
        for mc in MCs:
            k_nearest_MCs, k_nearest_dists = self._get_k_nearest_MCs(mc.center, k, contains_unlabeled=False)
            k_nearest_MCs = k_nearest_MCs.flatten()
            k_nearest_dists = k_nearest_dists.flatten()
            end_points = []
            for n_mc, dist in zip(k_nearest_MCs, k_nearest_dists):
                if n_mc.cls == mc.cls:
                    end_points.append(n_mc.center)
            k_c = len(end_points)

            if extra_sampling_num is None:
                # mc_sampling_num = max(1, self._random_state.poisson(coeff))
                mc_sampling_num = self._random_state.poisson((k - k_c) / 2) + 1
                # mc_sampling_num = self._random_state.poisson((len(end_points)) // 2) + 1
            else:
                mc_sampling_num = self._random_state.poisson(extra_sampling_num / k * (k - k_c)) + 1

            if len(end_points) == 0:
                end_points.append(mc.center)

            # Generate random vectors and calculate new samples
            random_coefs = 1 - self._random_state.rand(mc_sampling_num, len(end_points), 1)
            tmp_vectors = np.sum(random_coefs * (np.array(end_points) - mc.center), axis=1)
            norms = np.linalg.norm(tmp_vectors, axis=1, keepdims=False)
            dists = (1 - self._random_state.rand(mc_sampling_num, 1)) * mc.radius
            dirs = np.vstack(
                [vec / norm if norm != 0 else np.zeros_like(mc.center) for vec, norm in zip(tmp_vectors, norms)])
            new_samples = dists * dirs + mc.center
            new_labels = np.full((mc_sampling_num,), mc.cls)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(new_labels)

        # Concatenate samples and labels from all micro clusters
        samples = np.vstack(samples)
        labels = np.concatenate(labels)

        return samples, labels

    def sampling_by_cdi_based_KNN(self, MCs, MCS_truth, k, cdi_thr, extra_sampling_num=None):
        # Initialize empty lists for samples and labels
        samples = []
        labels = []

        # Iterate through all micro clusters
        for mc in MCs:
            k_nearest_MCs, k_nearest_dists = self._get_k_nearest_MCs(mc.center, k, contains_unlabeled=False)
            k_nearest_MCs = k_nearest_MCs.flatten()
            k_nearest_dists = k_nearest_dists.flatten()
            end_points = []
            end_points_cdis = []
            for n_mc, dist in zip(k_nearest_MCs, k_nearest_dists):
                if n_mc.cls == mc.cls:
                    end_points.append(n_mc.center)
                    cdi, _ = calc_cdi_based_on_MCS(MCS_truth, self, n_mc.center)
                    end_points_cdis.append(cdi)
            k_c = len(end_points)

            if extra_sampling_num is None:
                # mc_sampling_num = max(1, self._random_state.poisson(coeff))
                mc_sampling_num = self._random_state.poisson((k - k_c) / 2) + 1
                # mc_sampling_num = self._random_state.poisson((len(end_points)) // 2) + 1
            else:
                mc_sampling_num = self._random_state.poisson(extra_sampling_num / k * (k - k_c)) + 1

            if len(end_points) == 0:
                end_points.append(mc.center)

            end_points_cdis = np.array([(0 if cdi > cdi_thr else 1 - (cdi / cdi_thr)) for cdi in end_points_cdis])

            # Generate random vectors and calculate new samples
            weights = np.tile(np.expand_dims(end_points_cdis, axis=1), (mc_sampling_num, 1, 1))
            random_coefs = self._random_state.rand(mc_sampling_num, len(end_points), 1) * weights
            tmp_vectors = np.sum(random_coefs * (np.array(end_points) - mc.center), axis=1)
            norms = np.linalg.norm(tmp_vectors, axis=1, keepdims=False)

            dirs = np.vstack(
                [vec / norm if norm != 0 else np.zeros_like(mc.center) for vec, norm in zip(tmp_vectors, norms)])
            dists = (1 - self._random_state.rand(mc_sampling_num, 1)) * mc.radius
            new_samples = dists * dirs + mc.center
            new_labels = np.full((mc_sampling_num,), mc.cls)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(new_labels)

        # Concatenate samples and labels from all micro clusters
        samples = np.vstack(samples) if samples else None
        labels = np.concatenate(labels) if labels else None

        return samples, labels

    def sampling_from_MCs(self, MCs, sampling_num_for_one_MC=-1):
        # Initialize empty lists for samples and labels
        samples = []
        labels = []

        # Iterate through all micro clusters
        for mc in MCs:
            # Perform sampling in the current micro cluster
            new_samples, new_labels = self.sampling_from_MC(mc, sampling_num_for_one_MC)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(new_labels)

        # Concatenate samples and labels from all micro clusters
        samples = np.vstack(samples)
        labels = np.concatenate(labels)

        return samples, labels

    def sampling_from_new_and_removed_MCs(self, MCs, removed_MCs, sampling_num_for_one_MC=-1):
        # Initialize empty lists for samples and labels
        samples = []
        labels = []

        # Iterate through all micro clusters
        for mc, rmc in zip(MCs, removed_MCs):
            # Perform sampling in the current micro cluster
            new_samples, new_labels = self.sampling_from_MC(mc, sampling_num_for_one_MC)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(new_labels)

            # Perform sampling in the current micro cluster
            new_samples, new_labels = self.sampling_from_MC(rmc, rmc.n_samples)

            # Append new samples and labels to respective lists
            samples.append(new_samples)
            labels.append(np.ones_like(new_labels) * mc.cls)

        # Concatenate samples and labels from all micro clusters
        samples = np.vstack(samples)
        labels = np.concatenate(labels)

        return samples, labels

    def sampling_from_MC(self, MC, sampling_num_for_one_MC=-1):
        if sampling_num_for_one_MC <= 0:
            sampling_num_for_one_MC = MC.n_samples + 1

        return MC.do_sampling(num_sampling=sampling_num_for_one_MC,
                              random_state=self._random_state)

    def decay(self):
        """
        Perform decay operation on micro-clusters and remove those with weight below the threshold.
        Should be executed on each time step.
        """
        for cls, cls_mcs in self._MCs_repo.items():
            idx = len(cls_mcs) - 1
            while idx >= 0:
                mc = cls_mcs[idx]
                mc.decay(self.decay_factor)
                # If weight of micro cluster is lower than the threshold after decay,
                # delete it.
                if mc.weight <= self.weight_threshold:
                    self._delete_MC(mc)
                idx -= 1

    def predict_proba(self, X, k=3, mode="majority", return_neighbors=False):
        """
        Predicts the class probabilities for each input sample using the k-nearest neighbor ensemble method.
        The probabilities of an input sample is computed as the mean predicted class probabilities of the
        base estimators in the ensemble.

        Parameters:
        X : array-like or pd.DataFrame
            Input samples to predict class probabilities for.

        k : int, array-like, optional (default=3)
            Number(s) of nearest neighbor estimators to use for the ensemble prediction.

        mode : {'majority', 'weighted_majority', 'gravity'}, optional (default='majority')
            Mode of combining class probabilities from the k-nearest neighbors:
            - 'majority': Majority voting based on class labels (uniform weights).
            - 'weighted_majority': Weighted voting based on the neighbors' weights (if available).
            - 'gravity': Weighted voting based on the inverse of the squared distances to the neighbors.

        return_neighbors : bool, optional (default=False)
            Whether to return the nearest neighbors involved in classification.
            - True, return `(proba, nearest_neighbors)`
            - False, return `proba` only.

        Returns:
        numpy.ndarray
            An array of shape (n_samples, n_classes) containing the class probabilities for each sample.
        """
        X = check_array(X)

        tot_proba = np.zeros((X.shape[0], max(self.classes) + 1))

        nearest_MCs = None

        if isinstance(k, int):
            ks = [k]
        else:
            ks = check_array(k, ensure_2d=False).astype(int)
            ks = np.sort(ks)

        for k in ks:
            k_nearest_MCs, k_nearest_dists = self._get_k_nearest_MCs(X, k, contains_unlabeled=False)
            tot_proba += self._vote_proba(k_nearest_MCs, k_nearest_dists, mode)

            if k == ks[-1]:
                nearest_MCs = k_nearest_MCs

        proba = tot_proba / np.sum(tot_proba, axis=1, keepdims=True)
        return (proba, nearest_MCs) if return_neighbors else proba

    def predict(self, X, k=3, mode="majority", return_neighbors=False):
        """
        Predicts the class labels for each input sample using the k-nearest neighbor ensemble method.

        Parameters:
        X : array-like or pd.DataFrame
            Input samples to predict class labels for.

        k : int, array-like, optional (default=3)
            Number(s) of nearest neighbor estimators to use for the ensemble prediction.

        mode : {'majority', 'weighted_majority', 'gravity'}, optional (default='majority')
            Mode of combining class probabilities from the k-nearest neighbors:
            - 'majority': Majority voting based on class labels (uniform weights).
            - 'weighted_majority': Weighted voting based on the neighbors' weights (if available).
            - 'gravity': Weighted voting based on the inverse of the squared distances to the neighbors.

        return_neighbors : bool, optional (default=False)
            Whether to return the nearest neighbors involved in classification.
            - True, return `(predicted_labels, nearest_neighbors)`
            - False, return `predicted_labels` only.

        Returns:
        numpy.ndarray
            An array of shape (n_samples,) containing the predicted class labels for each sample.
        """
        result = self.predict_proba(X, k, mode, return_neighbors)
        nearest_MCs = None
        if return_neighbors:
            proba, nearest_MCs = result
        else:
            proba = result
        predicted_labels = np.argmax(proba, axis=1)

        return (predicted_labels, nearest_MCs) if return_neighbors else predicted_labels

    def _vote_proba(self, k_nearest_MCs, k_nearest_dists, mode):
        proba = np.zeros((len(k_nearest_MCs), max(self.classes) + 1))

        for i, MCs in enumerate(k_nearest_MCs):
            for j, mc in enumerate(MCs):
                if mc.cls == -1:
                    continue
                if mode == "majority":
                    vote_value = 1
                elif mode == "weighted_majority":
                    vote_value = mc.weight
                elif mode == "gravity":
                    vote_value = mc.weight / k_nearest_dists[i, j] ** 2
                else:
                    raise ValueError(f"Unsupported KNN mode: {mode}")
                proba[i, int(mc.cls)] += vote_value

        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def _get_k_nearest_MCs(self, X, k, contains_unlabeled=True):
        # Handle the case when k is 0 or negative
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        X = check_array(X)
        all_MCs = np.array(self.all_MCs(contains_unlabeled))
        all_centers = [mc.center for mc in all_MCs]

        # Ensure k is not greater than the total number of micro-clusters
        if len(all_MCs) < k:
            warnings.warn("Not enough micro clusters, so `k` is set as the number of micro clusters.")
            k = len(all_MCs)

        # Build a Ball Tree from the micro-cluster centers for efficient nearest neighbor search
        if self.n_features >= HIGH_DIM_THR:
            tree = BallTree(all_centers, metric="minkowski", p=1)
        else:
            tree = KDTree(all_centers, metric="minkowski", p=2)

        # Compute pairwise distances between X and all micro-cluster centers
        k_nearest_distances, k_nearest_indices = tree.query(X, k=k)

        # Find the k-nearest micro-clusters to the data points in X
        k_nearest_MCs = np.array([all_MCs[indices] for indices in k_nearest_indices])

        return k_nearest_MCs, k_nearest_distances

    def all_MCs(self, contains_unlabeled=True, to_numpy=False):
        all_MCs = [mc for cls, cls_mcs in self._MCs_repo.items() for mc in cls_mcs if contains_unlabeled or cls != -1]
        return np.array(all_MCs) if to_numpy else all_MCs

    def get_all_centers_and_weights(self, contains_unlabeled=True):
        all_MCs = self.all_MCs(contains_unlabeled)
        return np.vstack([MC.center for MC in all_MCs]), np.array([MC.weight for MC in all_MCs])

    def get_all_centers_and_numbers(self, contains_unlabeled=True):
        all_MCs = self.all_MCs(contains_unlabeled)
        return np.vstack([MC.center for MC in all_MCs]), np.array([MC.n_samples for MC in all_MCs])

    def get_info_for_calc_covariance_matrix(self, contains_unlabeled=True):
        all_MCs = self.all_MCs(contains_unlabeled)
        all_nums = np.array([MC.n_samples for MC in all_MCs])
        all_centers = np.vstack([MC.center for MC in all_MCs])
        all_sigmas = np.stack([MC.sigma for MC in all_MCs])
        return all_nums, all_centers, all_sigmas

    def get_all_radius_of_MCs(self, contains_unlabeled=True):
        all_MCs = self.all_MCs(contains_unlabeled)
        return np.array([MC.radius for MC in all_MCs])

    def _append_MC(self, mc):
        if mc in self._MCs_indices:
            print(f"MC exists! cls={mc.cls} {mc}")
            return
        self._MCs_indices[mc] = len(self._MCs_repo[mc.cls])
        self._MCs_repo[mc.cls].append(mc)

    def _delete_MC(self, mc):
        # Keep at least one micro cluster in each class
        if mc.cls != -1 and len(self._MCs_repo[mc.cls]) <= 1:
            return

        idx = self._MCs_indices[mc]
        self._MCs_repo[mc.cls].pop(idx)
        del self._MCs_indices[mc]
        # Update indices of follow-up MCs
        self._adjust_indices_after_removal(idx, mc.cls)

    def _adjust_indices_after_removal(self, start_idx, removed_cls):
        for o_idx in range(start_idx, len(self._MCs_repo[removed_cls])):
            o_mc = self._MCs_repo[removed_cls][o_idx]
            self._MCs_indices[o_mc] = o_idx

    def _labeling_unlabeled_MC(self, mc, cls):
        assert mc.cls == -1, "The input micro cluster is not unlabeled!"
        self._delete_MC(mc)
        mc.cls = cls
        self._append_MC(mc)

    def _create_new_MC_on_point(self, cur_t, x, y, radius, weight=1.0):
        while self.n_MCs >= self.max_n_MCs:
            self._on_n_MC_reaches_limit()

        new_MC = MicroCluster(cur_t, cls=y, classes=self.classes, maintain_cov=self.maintain_cov)
        new_MC.insert_sample(cur_t, x, y, weight)
        new_MC.in_range(y)
        new_MC.radius = radius

        self._append_MC(new_MC)

        return new_MC

    def _on_n_MC_reaches_limit(self):
        if len(self._MCs_repo[-1]) == 0:
            MC1, MC2 = self._find_closest_labeled_pair_from_largest_class()
        else:
            MC1, MC2 = self._find_closest_labeled_unlabeled_pair()

        # Delete MCs which are going to be merged. DO NOT CHANGE THE ORDER HERE
        self._delete_MC(MC2)
        self._delete_MC(MC1)

        # Merge two MCs and add them back into repo
        new_MC = MicroCluster.merge(MC1, MC2)
        self._append_MC(new_MC)

    def _find_closest_labeled_unlabeled_pair(self):
        unlabeled_MCs = self._MCs_repo[-1]
        unlabeled_MC_centers = [mc.center for mc in unlabeled_MCs]

        # Find the nearest labeled MC object for each MC in the unlabeled MCs
        nearest_labeled_MCs, nearest_dists = self._get_k_nearest_MCs(unlabeled_MC_centers, 1,
                                                                     contains_unlabeled=False)
        nearest_labeled_idx = np.argmin(nearest_dists)

        # Get the nearest labeled MC and the corresponding unlabeled MC
        the_labeled_MC = nearest_labeled_MCs[nearest_labeled_idx][0]
        the_unlabeled_MC = unlabeled_MCs[nearest_labeled_idx]

        return the_labeled_MC, the_unlabeled_MC

    def _find_closest_labeled_pair_from_largest_class(self):
        # Get the MCs of the largest class
        largest_class = max(self.classes, key=lambda cls: len(self._MCs_repo[cls]))
        largest_class_MCs = self._MCs_repo[largest_class]

        if len(largest_class_MCs) < 2:
            raise RuntimeError("Make sure the number of micro clusters of the largest class >= 2 !")

        largest_class_MC_centers = np.asarray([mc.center for mc in largest_class_MCs])

        # Calculate distance matrix between MC centers
        if self.n_features >= HIGH_DIM_THR:
            dist_matrix = cdist(largest_class_MC_centers, largest_class_MC_centers, "minkowski", p=1)
        else:
            dist_matrix = cdist(largest_class_MC_centers, largest_class_MC_centers, "minkowski", p=2)

        # Set diagonal elements to a large value to ignore self-distances
        np.fill_diagonal(dist_matrix, np.inf)

        # Get the nearest labeled MC object and the corresponding MC object from the largest class
        i, j = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        i, j = (i, j) if i < j else (j, i)
        MC1, MC2 = largest_class_MCs[i], largest_class_MCs[j]

        return MC1, MC2

    def _calc_involve_range_basis(self, X, y, mode="mean"):
        n_samples = len(X)
        if mode == "min":
            return min(
                np.linalg.norm(X[i] - X[j]) for i, j in itertools.combinations(range(n_samples), 2) if y[i] != y[j])
        elif mode == "mean":
            min_dists = np.ones((max(self.classes) + 1, max(self.classes) + 1)) * np.inf
            for i, j in itertools.combinations(range(n_samples), 2):
                if y[i] == y[j]:
                    continue
                dist = np.linalg.norm(X[i] - X[j])
                c1, c2 = (y[i], y[j]) if y[i] <= y[j] else (y[j], y[i])
                min_dists[c1][c2] = min(min_dists[c1][c2], dist)
            sum_dists, cnt_dists = 0, 0
            for i, j in itertools.product(range(max(self.classes) + 1), repeat=2):
                if min_dists[i][j] != np.inf:
                    sum_dists += min_dists[i][j]
                    cnt_dists += 1
            return sum_dists / cnt_dists
        elif mode == "std":
            return np.linalg.norm(np.std(X, axis=0))
        else:
            raise ValueError(f"Mode: {mode} is not supported.")
