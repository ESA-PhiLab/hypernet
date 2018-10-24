import numpy as np
from typing import List
from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets. Useful when loading patches
    of the dataset and combining them"""

    def __init__(self, datasets: List[Dataset]):
        data, labels = self.combine_datasets(datasets)
        super(ConcatDataset, self).__init__(data, labels)

    @staticmethod
    def combine_datasets(datasets: List[Dataset]) -> [np.ndarray, np.ndarray]:
        data = [dataset.get_data for dataset in datasets]
        labels = [dataset.get_labels for dataset in datasets]
        return np.vstack(data), np.hstack(labels)
