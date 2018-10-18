import numpy as np
from typing import List


class ConcatDataset:
    """Dataset to concatenate multiple datasets. Useful when loading patches
    of the dataset and combining them"""
    def __init__(self, datasets: List):
        self.data, self.labels = self.combine_datasets(datasets)

    @staticmethod
    def combine_datasets(datasets: List):
        data = [dataset.data for dataset in datasets]
        labels = [dataset.labels for dataset in datasets]
        return np.vstack(data), np.vstack(labels)
