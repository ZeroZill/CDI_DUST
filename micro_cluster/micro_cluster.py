import numpy as np
from sklearn.utils import check_random_state

from utils import check_array, randsphere_uniform, check_weights


class MicroCluster:

    def __init__(self, cur_t=None, cls=None, classes=None, maintain_cov=True):
        """
        MicroCluster object.

        Parameters:
            cur_t: float or None
                Current time step for the micro cluster update. If None, it means the class label is not assigned yet.

            cls: int or None
                Class label of the micro cluster. If None, it means the class label is not assigned yet.

            classes: List[int] or None
                All classes that might emerge. If None, there will be no record of counter table.
        """
        # -1 stands for unlabeled
        if cls is None:
            cls = -1
        self.linear_sum = 0.0  # Linear sum
        self.squared_sum = 0.0  # Squared sum
        self.n_samples = 0  # Number of samples
        self.weight = 1.0  # Weight
        self.update_time = cur_t  # Update time
        self.cls = cls  # Class of micro cluster

        self.center = None  # Center
        self.radius = 0.0  # Radius

        ############################ New added ############################
        self.classes = classes

        # Covariance matrix
        self.maintain_cov = maintain_cov
        if self.maintain_cov:
            self.sigma = 0.0

        # A table for recording the number of samples of all classes ever encountered
        self.counter_table = None
        self.counter_row = None

        # A table for recording the weights of all classes at runtime
        self.weight_table = None
        self.weight_row = None

        if self.classes is not None:
            self.counter_row = np.zeros(max(classes) + 1)
            self.weight_row = np.zeros(max(classes) + 1)

    def _calc_center(self):
        return self.linear_sum / self.n_samples

    def _calc_radius(self):
        return np.sqrt(
            np.sum(self.squared_sum / self.n_samples)
            - np.sum((self.linear_sum / self.n_samples) ** 2)
            + 1e-8  # A small value to avoid zero-value
        )

    def insert_sample(self, cur_t, samples, samples_label=-1, sample_weights=None):
        """
        Incrementally insert samples into the micro-cluster, ensuring they all belong to the same class.
        And if the micro-cluster already has a label, make sure the input samples' labels match it.

        Parameters:
            cur_t: int
                The current time step associated with the inserted samples.

            samples: array-like, shape (n_samples, n_features)
                The samples to be inserted into the micro-cluster.

            samples_label: int, optional (default=-1)
                The label of the samples. Only a single sample label is supported.

            sample_weights: None, int, float or array-like, shape (n_samples,), optional (default=1)
                The weight of each sample. If an integer is provided, the same weight is used for all samples.

        Returns:
            None
        """
        # Check class of samples
        if self.cls != -1 and samples_label != -1 and samples_label != self.cls:
            raise ValueError("Cannot insert samples of other class into this micro cluster.")
        # Check update time of samples
        assert self.update_time <= cur_t, \
            f"Invalid update time ({cur_t}), " \
            f"which is earlier than the update time of micro cluster ({self.update_time})."
        # Check and convert samples to a numpy array
        samples = check_array(samples)
        n_samples, n_dim = samples.shape

        # Intermediate parameters for updating covariance matrix
        mu_old_samples = self.linear_sum / self.n_samples if self.n_samples > 0 else 0.0
        mu_new_samples = np.mean(samples, axis=0)
        if n_samples == 1:
            sigma_new_samples = 0.0
        else:
            sigma_new_samples = np.cov(samples, rowvar=False, bias=True)

        sample_weights = check_weights(sample_weights, n_samples)

        # Update LS (Linear Sum), SS (Squared Sum), N (Number of samples), W (Total weight), T (Current time),
        # and CL (Sample label) based on the inserted samples
        self.linear_sum += np.sum(samples, axis=0)
        self.squared_sum += np.sum(samples ** 2, axis=0)
        self.n_samples += n_samples
        self.weight = max(self.weight, np.max(sample_weights))
        self.update_time = cur_t
        if samples_label != -1:
            self.cls = samples_label

        # Update covariance matrix
        mu = self.linear_sum / self.n_samples
        delta_mu_old = np.reshape(mu - mu_old_samples, (1, -1))
        delta_mu_new = np.reshape(mu - mu_new_samples, (1, -1))
        if self.maintain_cov:
            self.sigma = (self.n_samples - n_samples) / self.n_samples * (self.sigma + delta_mu_old.T @ delta_mu_old) + \
                         n_samples / self.n_samples * (sigma_new_samples + delta_mu_new.T @ delta_mu_new)

        # Calculate the center of the micro-cluster (C) and the radius (R) based on the updated LS and SS
        self.center = self._calc_center()
        self.radius = self._calc_radius()

    def in_range(self, sample_cls, n_samples=1):
        """
        New added
        """
        if self.classes and sample_cls != -1:
            self.counter_row[sample_cls] += n_samples
            self.weight_row[sample_cls] += n_samples

    def decay(self, decay_factor):
        """
        Apply exponential decay to the weight of the micro-cluster.

        Parameters:
            decay_factor: float
                The decay rate. The higher the decay rate, the faster it decays.
        """

        # Apply exponential decay to the weight of the micro-cluster
        self.weight *= decay_factor

        if self.classes:
            self.weight_row *= decay_factor

    def do_sampling(self, num_sampling, random_state=None):
        """
        Generate random samples within the hypersphere defined by the micro-cluster.

        Parameters:
            num_sampling: int
                The number of random samples to generate.

            random_state: int, numpy.random.RandomState or None, optional (default=None)
                Random generator for sampling.

        Returns:
            numpy.ndarray: An array containing the randomly sampled data points within the hypersphere.
        """
        random_state = check_random_state(random_state)
        dim = len(self.center)
        samples = self.center + randsphere_uniform(num_sampling, dim, random_state) * self.radius
        labels = np.ones(num_sampling) * self.cls
        return samples, labels

    @staticmethod
    def merge(mc1, mc2):
        """
        Merge two MicroCluster objects into a new MicroCluster.

        Specifically, this method creates a new MicroCluster object by combining the attributes of mc1 and mc2.
            - The linear sum, squared sum, number of samples, weight, and update time are added together.
            - The maximum weight and update time are selected.
            - The class label is determined based on the following conditions:
                1. If both mc1 and mc2 have assigned class labels (not None), they must match.
                2. If only one of mc1 and mc2 has a class label, the class label of that micro cluster is used.
                3. If neither mc1 nor mc2 has an assigned class label (both are None), the class label will be None.
            - The center and radius of the merged MicroCluster are calculated based on its updated attributes.

        Parameters:
            mc1: MicroCluster
                The first MicroCluster to be merged.
            mc2: MicroCluster
                The second MicroCluster to be merged.

        Returns:
            MicroCluster: A new MicroCluster object that is the result of merging mc1 and mc2.
        """
        assert mc1.cls == -1 or mc2.cls == -1 or mc1.cls == mc2.cls, \
            "If both micro clusters have assigned class labels (not -1), they must match."

        # Use the closest update time
        update_time = max(mc1.update_time, mc2.update_time)
        # Determine the class label of the merged MicroCluster based on the conditions mentioned in the Note.
        cls = mc2.cls if mc1.cls is None else mc1.cls
        # Set classes
        classes = mc2.classes if mc1.classes is None else mc2.classes
        # Create a new MicroCluster object for the merged result
        merged_mc = MicroCluster(cur_t=update_time, cls=cls, classes=classes)

        # Combine the linear sum, squared sum, number of samples, weight, and update time of mc1 and mc2
        merged_mc.linear_sum = mc1.linear_sum + mc2.linear_sum
        merged_mc.squared_sum = mc1.squared_sum + mc2.squared_sum
        merged_mc.n_samples = mc1.n_samples + mc2.n_samples
        merged_mc.weight = max(mc1.weight, mc2.weight)

        # Update covariance matrix
        mu = merged_mc.linear_sum / merged_mc.n_samples if merged_mc.n_samples > 0 else 0.0
        mu_1 = mc1.linear_sum / mc1.n_samples if mc1.n_samples > 0 else 0.0
        mu_2 = mc2.linear_sum / mc2.n_samples if mc2.n_samples > 0 else 0.0
        delta_mu_1 = np.reshape(mu - mu_1, (1, -1))
        delta_mu_2 = np.reshape(mu - mu_2, (1, -1))
        if mc1.maintain_cov and mc2.maintain_cov:
            merged_mc.maintain_cov = True
            merged_mc.sigma = mc1.n_samples / merged_mc.n_samples * (mc1.sigma + delta_mu_1.T @ delta_mu_1) + \
                              mc2.n_samples / merged_mc.n_samples * (mc2.sigma + delta_mu_2.T @ delta_mu_2)

        # Calculate the center and radius of the merged MicroCluster based on its updated attributes
        merged_mc.center = merged_mc._calc_center()
        merged_mc.radius = merged_mc._calc_radius()

        # new added
        if merged_mc.classes:
            merged_mc.counter_row = mc1.counter_row + mc2.counter_row
            merged_mc.weight_row = mc1.weight_row + mc2.weight_row
        return merged_mc
