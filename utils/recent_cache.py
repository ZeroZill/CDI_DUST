import heapq

import torch


class NormalCache:
    def __init__(self):
        self.cache = {}

    def reset(self):
        self.cache = {}

    def insert(self, index, value):
        self.cache[index] = value

    def query(self, index):
        return self.__getitem__(index)

    def pop(self, index):
        if index not in self.cache:
            return
        self.cache.pop(index)

    def __getitem__(self, index):
        if index not in self.cache:
            raise IndexError(f"Index {index} not in cache.")

        return self.cache[index]

    def __contains__(self, index):
        return index in self.cache

    def get_samples_and_labels_as_tensor(self, length=-1):
        samples = []
        labels = []

        start_idx = max(0, len(self.cache) - length) if length > 0 else 0

        for _, (sample, label) in list(self.cache.items())[start_idx:]:
            samples.append(sample)
            labels.append(label)

        if samples:
            if isinstance(samples[0], torch.Tensor):
                samples_tensor = torch.stack(samples, dim=0)
                labels_tensor = torch.stack(labels, dim=0)
            else:
                samples_tensor = torch.tensor(samples)
                labels_tensor = torch.tensor(labels)
            return samples_tensor, labels_tensor
        else:
            return None, None


class MostRecentCache(NormalCache):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity
        self.index_heap = []

    def reset(self):
        super(MostRecentCache, self).reset()
        self.index_heap = []

    def insert(self, index, value):
        if index in self.cache:
            self.cache[index] = value
            return

        if len(self.cache) >= self.capacity:
            if self.index_heap and index < self.index_heap[0]:
                return

            min_index = heapq.heappop(self.index_heap)
            self.cache.pop(min_index)

        self.cache[index] = value
        heapq.heappush(self.index_heap, index)

    def query(self, index):
        return self.__getitem__(index)

    def pop(self, index):
        if index not in self.cache:
            return

        self.cache.pop(index)
        self.index_heap.remove(index)
        heapq.heapify(self.index_heap)
