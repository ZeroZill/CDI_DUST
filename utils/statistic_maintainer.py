import torch


class AverageStatistic:
    def __init__(self):
        self.mean = 0.0
        self.cnt = 0

    def update(self, x, delta_cnt=1):
        self.mean = (self.mean * self.cnt + x * delta_cnt) / (self.cnt + delta_cnt)
        self.cnt += delta_cnt

    def reset(self):
        self.mean = 0.0
        self.cnt = 0

    def __lt__(self, other):
        return self.mean < other.mean

    def __gt__(self, other):
        return self.mean > other.mean

    def __eq__(self, other):
        return self.mean == other.mean


class Statistics:
    def __init__(self):
        self.mean = 0.0
        self.std = 0.0
        self.var = 0.0
        self.cnt = 0

    def update(self, x):
        self.cnt += 1
        old_mean = self.mean
        self.mean += (x - self.mean) / self.cnt
        self.var += (x - self.mean) * (x - old_mean)
        self.std = torch.sqrt(self.var / self.cnt)

    def reset(self):
        self.mean = 0.0
        self.std = 0.0
        self.var = 0.0
        self.cnt = 0
