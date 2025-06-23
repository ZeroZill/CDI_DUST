import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import chi2
from sklearn.utils import check_random_state

from data.jit_sdp_stream_loader import JITSDPStreamLoader
from sliding_window import SlidingWindow
from stability_indicators.cdi import calc_cdi_based_on_MCS, get_cdi_of_new_point
from data.delayed_stream_loader import DelayedStreamLoader
from utils.data_utils import normalization
from utils.running_records import RunningRecords


def calc_cdi_threshold(cdi, inc_threshold, num_classes, function="exp", beta=1.0):
    x = min(np.abs(cdi), inc_threshold)
    T = inc_threshold
    c = num_classes
    if function == "linear":
        return ((c - beta) * x + beta * T) / (c * T)
    elif function == "exp":
        return (c / beta) ** (x / T - 1)
    elif function == "pow":
        return beta / c + (1 - beta / c) * (x / T) ** 0.5
    else:
        raise ValueError(f"Invalid function type {function}")


def run_stream_dust(
        classifier, micro_clustering_system, X, Y,
        delay=500, n_pretrain=500,
        k=50,
        mode="dust",  # dust, wait, topline
        seed=0,
        do_normalization=True,
        k_in_knn=7,
        cdi_alpha=0.05,
        dataset=None,
        commit_ts=None,
        show_plot=False,
        verbose=0,  # 0: no, 1: only print result, 2: print key information, >=3: print detailed information
):
    torch.manual_seed(seed)
    random_state = check_random_state(seed)

    running_records = RunningRecords(
        "test_idx",
        "y_true",
        "y_pred",
        "y_pseudo",
        "accumulating_accs_y_pred",
        "accumulating_accs_y_pseudo",
        "used_pseudo"
    )

    classes = np.unique(Y)

    mode = mode.lower()
    if "dust" in mode:
        MCS_truth = copy.deepcopy(micro_clustering_system)
        MCS_unlabeled = copy.deepcopy(micro_clustering_system)
        MCS_truth.set_seed(random_state)
        MCS_unlabeled.set_seed(random_state)

    # Calculate the CDI threshold
    cdi_thr = chi2.ppf(1 - cdi_alpha, X.shape[1])

    if verbose >= 1:
        print(f"\n========== [Mode] <{mode.upper()}>-delay_{np.mean(delay)} ==========")

    ''' Normalization '''
    if do_normalization:
        X = normalization(X)

    if dataset is not None and "JIT-SDP" in dataset:
        assert isinstance(delay, np.ndarray), "For JIT-SDP, delay must be an array."
        stream = JITSDPStreamLoader(X, Y, commit_ts, delay, n_pretrain=n_pretrain)
    else:
        stream = DelayedStreamLoader(X, Y, delays=delay, n_pretrain=n_pretrain)

    ''' Pretrain '''
    if n_pretrain is not None and n_pretrain > 0:
        start = time.time()
        if verbose >= 2:
            print("[Initial phase]\tStart!")
        initial_X, initial_Y = stream.fetch_initial_samples()
        classifier.partial_fit(initial_X, initial_Y, classes=classes)
        if "dust" in mode:
            MCS_truth.initialize_MCs(initial_X, initial_Y, k)
            MCS_unlabeled.initialize_MCs(initial_X, initial_Y, k)

            MCS_truth.set_time(stream.current_time_step())
            MCS_unlabeled.set_time(stream.current_time_step())
        if verbose >= 2:
            print("[Initial phase]\tOver! Time consumed: {} (s)".format(time.time() - start))

    if verbose >= 2:
        print("[Online phase]\tStart!")
    start = time.time()
    # Running
    while stream.has_next():
        ''' Showing progress '''
        if verbose >= 3:
            print("\r[Online phase]\tProgress: {}/{}".format(stream.current_time_step(), stream.total_time_step()),
                  end='',
                  flush=True)

        test_index, test_x, test_y, \
        train_index, train_x, train_y = stream.next_test_and_train_samples()

        pseudo_idx, pseudo_x, pseudo_y = None, None, None

        ''' Testing '''
        for idx, x, y in zip(test_index, test_x, test_y):
            y_proba = classifier.predict_proba([x])
            y_hat = np.argmax(y_proba, axis=1)

            running_records.add_record("test_idx", idx)
            running_records.add_record("y_true", y)
            running_records.add_record("y_pred", y_hat[0])
            running_records.add_record("accumulating_accs_y_pred", float(y == y_hat[0]))

            if "dust" in mode:
                y_m_proba = MCS_truth.predict_proba([x], k=k_in_knn)
                y_m = np.argmax(y_m_proba, axis=1)

                running_records.add_record("y_pseudo", y_m[0])
                running_records.add_record("accumulating_accs_y_pseudo", float(y_m[0] == y))

                # if corresponding label not arrive immediately, use pseudo labels
                if train_index is None or idx not in train_index:
                    cdi, offset = calc_cdi_based_on_MCS(MCS_truth, MCS_unlabeled, x)
                    if np.max(y_m_proba) >= calc_cdi_threshold(cdi, cdi_thr, max(classes) + 1):
                        if pseudo_x is None:
                            pseudo_idx, pseudo_x, pseudo_y = np.array([idx]), np.array([x]), np.array(y_m)
                        else:
                            pseudo_idx = np.concatenate((pseudo_idx, idx))
                            pseudo_x = np.vstack((pseudo_x, x))
                            pseudo_y = np.concatenate((pseudo_y, y_m))

                        running_records.add_record("used_pseudo", 1)

                MCS_unlabeled.partial_fit(x, y_m, is_pseudo=True)

        ''' Training '''
        if "dust" in mode:
            if train_x is not None and train_y is not None:
                for idx, x, y in zip(train_index, train_x, train_y):
                    MCS_truth.partial_fit([x], [y], is_pseudo=False)

                    cdi, offset = calc_cdi_based_on_MCS(MCS_truth, MCS_unlabeled, x)
                    if cdi <= cdi_thr:
                        classifier.partial_fit([x], [y])
                    classifier.partial_fit([x], [y])

            if "wo_ip" not in mode:
                if pseudo_x is not None and pseudo_y is not None:
                    for px, py in zip(pseudo_x, pseudo_y):
                        classifier.partial_fit([px], [py])
        elif mode == "wait":
            if train_x is not None and train_y is not None:
                classifier.partial_fit(train_x, train_y)
        elif mode == "topline":
            classifier.partial_fit(test_x, test_y)

        # ''' Fading '''
        if "dust" in mode:
            MCS_truth.set_time(stream.current_time_step())
            MCS_unlabeled.set_time(stream.current_time_step())

        running_records.fill_blank_records()

    if verbose >= 2:
        print("\n[Online phase]\tOver! Time consumed: {} (s)".format(time.time() - start))

    if verbose >= 1:
        print("[Final accuracy] ", running_records["accumulating_accs_y_pred"][-1])

    if show_plot:
        plot_res(dataset, mode, running_records)

    return running_records


def plot_res(dataset, mode, running_records):
    len_rec = len(running_records["accumulating_accs_y_pred"])
    plt.plot(np.arange(len_rec), running_records["accumulating_accs_y_pred"], label="Accuracy")
    if "dust" in mode:
        print("[Final accuracy MC] ", running_records["accumulating_accs_y_pseudo"][-1])

        plt.plot(np.arange(len_rec), running_records["accumulating_accs_y_pseudo"], label="Accuracy_mc")
    plt.title(f"{mode.upper()}: {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.show()
