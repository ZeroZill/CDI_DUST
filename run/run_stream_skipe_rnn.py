import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from torch.nn import functional as F

from base_models.skipe_rnn import SkipERNNModel
from data.delayed_stream_loader import DelayedStreamLoader
from data.jit_sdp_stream_loader import JITSDPStreamLoader
from utils import normalization, RunningRecords


def run_stream_skipe_rnn(X, Y,
                         n_classes,
                         delay=500, n_pretrain=500,
                         eta=0.5,
                         lambd=0.001,
                         lr=0.01,
                         momentum=0.95,
                         seed=0,
                         dataset=None,
                         commit_ts=None,
                         do_normalization=True,
                         show_plot=True,
                         verbose=0,
                         # 0: no, 1: only print result, 2: print key information, >=3: print detailed information
                         ):
    """
    Run experiment on SkipE-RNN.
    """
    # Records to be kept
    running_records = RunningRecords(
        "y_true",
        "y_pred",
        "accumulating_accs_y_pred"
    )
    random_state = check_random_state(seed)
    torch.manual_seed(seed)

    n_features, n_dim = X.shape

    # To tensor
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

    # Change labels to one-hot encoding
    Y = F.one_hot(Y, n_classes)

    y_kt = Y[0]
    y_kt_true = None

    skipe_rnn = SkipERNNModel(input_size=n_dim, output_size=n_classes,
                              eta=eta, activate="hyperplane",
                              lambd=lambd,
                              lr=lr, momentum=momentum)

    if verbose >= 1:
        print(f"\n========== [SkipE-RNN] delay_{np.mean(delay)} ==========")

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
        skipe_rnn.initial_fit(initial_X, initial_Y)
        y_kt = initial_Y[-1]
        y_kt_true = initial_Y[-1]
        if verbose >= 2:
            print("[Initial phase]\tOver! Time consumed: {} (s)".format(time.time() - start))

    fading_wrong_rates = []
    fading_factor_wrong_rate = 0.99

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
            y_hat, y_kt = skipe_rnn.predict(x, y_kt)

            y_true = torch.argmax(y, dim=-1).item()
            y_pred = y_hat.item()
            running_records.add_record("y_true", y_true)
            running_records.add_record("y_pred", y_pred)
            running_records.add_record("accumulating_accs_y_pred", float(y_pred == y_true))

            ''' Pre-Training '''
            skipe_rnn.partial_fit(x, y, idx, y_avail=False)

            if len(fading_wrong_rates) == 0:
                fading_wrong_rates.append(y_hat != y)
            else:
                fading_wrong_rates.append(
                    fading_wrong_rates[-1] * fading_factor_wrong_rate + (y_hat != y) * (1 - fading_factor_wrong_rate))

        ''' Training '''
        if train_x is not None and train_y is not None:
            for idx, x, y in zip(train_index, train_x, train_y):
                skipe_rnn.partial_fit(x, y, idx, y_avail=True, last_y=y_kt_true)
                y_kt_true = y

        running_records.fill_blank_records()

    if verbose >= 2:
        print("\n[Online phase]\tOver! Time consumed: {} (s)".format(time.time() - start))

    if verbose >= 1:
        print("[Final accuracy] ", running_records["accumulating_accs_y_pred"][-1])

    if show_plot:
        plt.plot(np.arange(len(running_records["accumulating_accs_y_pred"])),
                 running_records["accumulating_accs_y_pred"], label="Accuracy")

        plt.title("SkipE-RNN")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return running_records
