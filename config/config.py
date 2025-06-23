EXP_DATASETS = {
    # <dataset> : (<n_features>, <n_classes>, <n_samples>)
    # <n_samples> is `None` if the dataset is a pre-defined one.
    # ==============================================================
    # Synthetic data
    "Inc_Hyperplane_1": (10, 2, 10000),
    "Inc_Hyperplane_2": (10, 2, 10000),
    "Inc_Hyperplane_3": (10, 2, 10000),
    "Inc_RBF_1": (10, 2, 10000),
    "Inc_RBF_2": (10, 2, 10000),
    "Inc_RBF_3": (10, 2, 10000),
    "SEA_r": (3, 2, 16000),
    "SEA_n": (3, 2, 16000),
    "Agrawal_r": (9, 2, 16000),
    "Agrawal_n": (9, 2, 16000),
    "LED_r": (24, 10, 16000),
    "LED_n": (24, 10, 16000),
    "Sine_r": (2, 2, 16000),
    "Sine_n": (2, 2, 16000),
    "SEA_rg": (3, 2, 16000),
    "SEA_ng": (3, 2, 16000),
    "Agrawal_rg": (9, 2, 16000),
    "Agrawal_ng": (9, 2, 16000),
    "LED_rg": (24, 10, 16000),
    "LED_ng": (24, 10, 16000),
    "Sine_rg": (2, 2, 16000),
    "Sine_ng": (2, 2, 16000),
    # ==============================================================
    "Electricity": (8, 2, None),  # 45312
    "Weather": (8, 2, None),  # 18159
    "Airlines": (7, 2, None),  # 539383
    "Covtype": (54, 7, 50000),  # 581012
    "Rialto": (27, 10, None),  # 82250
    "INS-Inc-Reo": (33, 6, None),  # 79986, (change points: 26568, 53364)
    "INS-Inc-Abt": (33, 6, None),  # 79986, (change points: 26568, 53364)
    "INS-Grad": (33, 6, None),  # 24150, (change points: 14028)
    "Poker_hand": (10, 10, 100000),  # 1025010
    "Asfault": (64, 6, None),  # 8564
    "UWave": (315, 8, None),  # 4478
    "JIT-SDP_django": (14, 2, None),  # 31376
    "JIT-SDP_pandas": (14, 2, None),  # 31138
    # ==============================================================
}
