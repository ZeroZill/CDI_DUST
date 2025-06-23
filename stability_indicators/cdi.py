import numpy as np
from sklearn.neighbors import KDTree, BallTree

HIGH_DIM_THR = 30


def covariance_for_MCs(points, nums, sigmas, unbiased=True):
    K = np.sum(nums)
    mu = np.average(points, axis=0, weights=nums)

    deltas = points - mu
    weighted_covariances = nums[:, np.newaxis, np.newaxis] * \
                           (sigmas + np.einsum('ij,ik->ijk', deltas, deltas))
    sigma = np.sum(weighted_covariances, axis=0) / K

    if unbiased:
        sigma *= K / (K - 1)

    return sigma


def get_cdi_of_new_point(k, point,
                         points_ref, points_cur,
                         for_MC=False,
                         nums_ref=None, nums_cur=None,
                         sigmas_ref=None, sigmas_cur=None
                         ):
    if for_MC:
        assert len(points_ref) == len(nums_ref) == len(sigmas_ref), \
            "Length of input parameters of ref window must be the same."

        assert len(points_cur) == len(nums_cur) == len(sigmas_cur), \
            "Length of input parameters of cur window must be the same."

    # Construct KD trees for reference and current points
    if points_ref.shape[1] < HIGH_DIM_THR:
        ref_tree = KDTree(points_ref, metric="minkowski", p=2)
        cur_tree = KDTree(points_cur, metric="minkowski", p=2)
    else:
        ref_tree = BallTree(points_ref, metric="minkowski", p=1)
        cur_tree = BallTree(points_cur, metric="minkowski", p=1)

    # Find k nearest reference points
    nearest_ref_distances, nearest_ref_indices = ref_tree.query(point, k=k)
    nearest_ref_indices = nearest_ref_indices[0]

    # Determine radius for current points
    max_ref_distance = np.max(nearest_ref_distances)

    # Find nearest current points within the determined radius
    nearest_cur_indices = cur_tree.query_radius(point, r=max_ref_distance)
    nearest_cur_indices = nearest_cur_indices[0]

    # Get coordinates of nearest reference and current points
    nearest_ref_points = points_ref[nearest_ref_indices]
    nearest_cur_points = points_cur[nearest_cur_indices]

    # Calculate weighted covariance matrix and offset between nearest current and reference points
    if not for_MC:
        sigma_W = np.cov(np.vstack((nearest_ref_points, nearest_cur_points)), rowvar=False, bias=False)
        if len(nearest_cur_indices) > 0 and len(nearest_ref_indices) > 0:
            offset = np.mean(nearest_cur_points, axis=0) - np.mean(nearest_ref_points, axis=0)
        else:
            offset = np.ones_like(point) * np.inf

        s_nearest_ref = len(nearest_ref_indices)
        s_nearest_cur = len(nearest_cur_indices)
    else:
        nearest_ref_nums = nums_ref[nearest_ref_indices]
        nearest_cur_nums = nums_cur[nearest_cur_indices]
        nearest_ref_sigmas = sigmas_ref[nearest_ref_indices]
        nearest_cur_sigmas = sigmas_cur[nearest_cur_indices]
        sigma_W = covariance_for_MCs(
            np.vstack((nearest_ref_points, nearest_cur_points)),
            np.concatenate((nearest_ref_nums, nearest_cur_nums)),
            np.concatenate((nearest_ref_sigmas, nearest_cur_sigmas), axis=0),
            unbiased=True
        )
        if len(nearest_cur_indices) > 0 and len(nearest_ref_indices) > 0:
            offset = np.average(nearest_cur_points, axis=0, weights=nearest_cur_nums) \
                     - np.average(nearest_ref_points, axis=0, weights=nearest_ref_nums)
        else:
            offset = np.ones_like(point) * np.inf

        s_nearest_ref = np.sum(nearest_ref_nums)
        s_nearest_cur = np.sum(nearest_cur_nums)

    # Calculate CDI
    if np.isinf(offset).any():
        m2 = np.inf
    else:
        try:
            inv = np.linalg.inv(sigma_W)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sigma_W)
        m2 = (s_nearest_ref * s_nearest_cur / (s_nearest_ref + s_nearest_cur)) * \
             offset.T @ inv @ offset

    return offset, m2


def calc_cdi_based_on_MCS(MCS_ref, MCS_cur, x):
    nums_cur, centers_cur, sigmas_cur = MCS_cur.get_info_for_calc_covariance_matrix()
    nums_ref, centers_ref, sigmas_ref = MCS_ref.get_info_for_calc_covariance_matrix()
    k_in_cdi = int(len(centers_ref) * 0.1)
    offset, cdi = get_cdi_of_new_point(k_in_cdi, x.reshape(1, -1),
                                       centers_ref, centers_cur,
                                       for_MC=True,
                                       nums_ref=nums_ref, nums_cur=nums_cur,
                                       sigmas_ref=sigmas_ref, sigmas_cur=sigmas_cur)
    return cdi, offset
