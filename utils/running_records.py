import pandas as pd


class RunningRecords:
    def __init__(self, *keys):
        """
        Initializes the running records.

        :param keys: Items to be recorded. For accumulating items, include "accumulating" in the name.
        """
        self.records = {key: [] for key in keys}

    def __getitem__(self, item):
        return self.records[item]

    def add_record(self, key, value):
        if key not in self.records:
            raise KeyError(f"No such recording item {key}")
        self._add_record(key, value, accumulating="accumulating" in key)

    def add_empty_record(self, key):
        if key not in self.records:
            raise KeyError(f"No such recording item {key}")
        self._add_record(key, None, accumulating="accumulating" in key)

    def _add_record(self, key, value, accumulating=False):
        if accumulating:
            self._add_accumulating_record(key, value)
        else:
            self._add_normal_record(key, value)

    def _add_normal_record(self, key, value):
        self.records[key].append(value)

    def _add_accumulating_record(self, key, value):
        n = len(self.records[key])
        if value is None:
            value = self.records[key][-1] if n > 0 else 0.0
        if n == 0:
            self.records[key].append(value)
        else:
            accumulated_value = (n / (n + 1) * self.records[key][-1] + 1 / (n + 1) * value)
            self.records[key].append(accumulated_value)

    def fill_blank_records(self):
        """
        Fills all records to the length of the longest record. Each record is filled with empty values
        according to whether it is an accumulating record or not.
        """
        # Find the maximum length of all records
        max_length = max(len(record) for record in self.records.values())

        # Fill each record to the maximum length
        for key, record in self.records.items():
            while len(record) < max_length:
                self.add_empty_record(key)


def analyze_conflicts_from_records(detailed_res: pd.DataFrame):
    # Get all non-None values from "conflicted_idx" column
    conflicted_indices = detailed_res['conflicted_idx'].dropna().tolist()

    # Find all rows where "test_idx" is in conflicted_indices
    conflicted_data = detailed_res[detailed_res['test_idx'].isin(conflicted_indices)]

    # Identify hits where y_true != y_pseudo and used_pseudo == 1
    hits = conflicted_data[(conflicted_data['y_true'] != conflicted_data['y_pseudo']) &
                           (conflicted_data['used_pseudo'] == 1)]

    # Calculate number of hits and conflicts
    num_hits = len(hits)
    num_conflicts = len(conflicted_data)

    # Calculate hit rate
    hit_rate = num_hits / num_conflicts if num_conflicts > 0 else 0

    # Find total relevant rows in the original data where y_true != y_pseudo and used_pseudo == 1
    total_relevant = len(detailed_res[(detailed_res['y_true'] != detailed_res['y_pseudo']) &
                                      (detailed_res['used_pseudo'] == 1)])

    # Calculate recall rate
    recall_rate = num_hits / total_relevant if total_relevant > 0 else 0

    return hit_rate, recall_rate
