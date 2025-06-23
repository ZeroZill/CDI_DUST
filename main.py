"""
Here is a demo of how to use DUST
"""
import time

import numpy as np
from skmultiflow.meta import AdaptiveRandomForestClassifier, OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier

from micro_cluster import MicroClusterSystem
from run import run_stream_dust, run_stream_miplosc
from utils import load_jit_sdp_dataset, load_dataset, generate_delays

seed = 42
dataset = "JIT-SDP_django"
n_pretrain = 500
delay_mode = "fixed"  # "fixed" or "varying"
delay = 1000

# Default parameters
para_comb = {
    'n_clusters_per_class': 50,
    'max_n_MCs': 1000,
    'decay_factor': 0.9977,
    'weight_threshold': 0.01,
    'mc_threshold': 1.0,
    'k_in_knn': 7,
    'lr': 0.01
}

if "JIT-SDP" in dataset:
    X, Y, unix_timestamps, stream_delay = load_jit_sdp_dataset(dataset)
else:
    X, Y = load_dataset(dataset=dataset)
    unix_timestamps = None
    stream_delay = generate_delays(delay_mode, delay, len(X) - n_pretrain)

n_classes = len(np.unique(Y))

print("Start!")
start_time = time.time()
clf = AdaptiveRandomForestClassifier(split_criterion="gini", random_state=seed)
mcs = MicroClusterSystem(list(range(n_classes)),
                         max_n_MCs=para_comb['max_n_MCs'],
                         decay_factor=para_comb['decay_factor'],
                         weight_threshold=para_comb['weight_threshold'],
                         conflict_mode="dominate",
                         random_state=seed)

running_records = run_stream_dust(clf, mcs, X, Y,
                                  delay=stream_delay,
                                  n_pretrain=n_pretrain,
                                  k=para_comb['n_clusters_per_class'],
                                  k_in_knn=para_comb['k_in_knn'],
                                  seed=seed,
                                  dataset=dataset,
                                  commit_ts=unix_timestamps,
                                  verbose=3,
                                  show_plot=False)
# clf = OzaBaggingClassifier(base_estimator=HoeffdingTreeClassifier(split_criterion="gini"),
#                                    random_state=seed)
# cwmc = MicroClusterSystem(list(range(n_classes)),
#                                   max_n_MCs=para_comb['max_n_MCs'],
#                                   decay_factor=para_comb['decay_factor'],
#                                   weight_threshold=para_comb['weight_threshold'],
#                                   conflict_mode="dominate",
#                                   maintain_cov=False,
#                                   random_state=seed)
#
# running_records = run_stream_miplosc(clf, cwmc, X, Y,
#                                      delay=stream_delay,
#                                      n_pretrain=n_pretrain,
#                                      k=para_comb['n_clusters_per_class'],
#                                      k_in_knn=para_comb['k_in_knn'],
#                                      seed=seed,
#                                      dataset=dataset,
#                                      commit_ts=unix_timestamps,
#                                      verbose=3,
#                                      show_plot=False)
time_cost = time.time() - start_time
final_acc = running_records["accumulating_accs_y_pred"][-1]
print(f"Over! Acc: {final_acc}, Time: {time_cost}")
