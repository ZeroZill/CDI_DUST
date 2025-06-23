import time

import matplotlib.pyplot as plt
import numpy as np

from base_models.osnn import OSNN
from data.delayed_stream_loader import DelayedStreamLoader
from data.jit_sdp_stream_loader import JITSDPStreamLoader
from utils.data_utils import normalization
from utils.running_records import RunningRecords


def run_stream_osnn(
        X, Y,
        delay=500, n_pretrain=500,
        num_center=10,
        window_size=200,
        beta=1,
        gamma=1,
        optim_steps=10,
        use_unlabeled_data=True,
        do_normalization=True,
        seed=0,
        dataset=None,
        commit_ts=None,
        show_plot=True,
        verbose=0,  # 0: no, 1: only print result, 2: print key information, >=3: print detailed information
):
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

    if len(classes) != 2:
        if verbose >= 1:
            print("[WARNING] OSNN can only be used for binary classification.")
        return running_records

    classifier = OSNN(num_center=num_center,
                      window_size=window_size,
                      beta=beta,
                      gamma=gamma,
                      optim_steps=optim_steps,
                      seed=seed)

    if verbose >= 1:
        print(f"\n========== [Mode] <OSNN>-delay_{np.mean(delay)} ==========")

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
        for x, y in zip(initial_X, initial_Y):
            classifier.train(x, y)
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

        ''' Testing '''
        for idx, x, y in zip(test_index, test_x, test_y):
            y_proba = classifier.predict_proba(x)
            y_hat = np.argmax(y_proba, axis=1)

            running_records.add_record("test_idx", idx)
            running_records.add_record("y_true", y)
            running_records.add_record("y_pred", y_hat[0])
            running_records.add_record("accumulating_accs_y_pred", float(y == y_hat[0]))

            # train on temporarily unlabeled data
            if use_unlabeled_data:
                classifier.train_on_unlabeled(x)

        ''' Training '''
        if train_x is not None and train_y is not None:
            for x, y in zip(train_x, train_y):
                classifier.train(x, y)

        running_records.fill_blank_records()

    if verbose >= 2:
        print("\n[Online phase]\tOver! Time consumed: {} (s)".format(time.time() - start))

    # acc = accuracy_score(y_pred, y_true)
    if verbose >= 1:
        print("[Final accuracy] ", running_records["accumulating_accs_y_pred"][-1])

    if show_plot:
        plot_res(dataset, "OSNN", running_records)

    return running_records


def plot_res(dataset, mode, running_records):
    len_rec = len(running_records["accumulating_accs_y_pred"])
    plt.plot(np.arange(len_rec), running_records["accumulating_accs_y_pred"], label="Accuracy")
    if "miplosc" in mode:
        print("[Final accuracy MC] ", running_records["accumulating_accs_y_pseudo"][-1])

        plt.plot(np.arange(len_rec), running_records["accumulating_accs_y_pseudo"], label="Accuracy_mc")
    plt.title(f"{mode.upper()}: {dataset}")
    plt.legend()
    plt.tight_layout()
    plt.show()
