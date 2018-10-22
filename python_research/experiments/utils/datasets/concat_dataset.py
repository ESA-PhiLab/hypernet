import numpy as np
from typing import List, Iterable
from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets. Useful when loading patches
    of the dataset and combining them"""

    def __init__(self, datasets: List[Dataset]):
        self.data, self.labels = self.combine_datasets(datasets)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        sample_x = self.data[item, ...]
        sample_y = self.labels[item, ...]
        return sample_x, sample_y

    def delete_by_indices(self, indices: Iterable):
        np.delete(self.data, indices)
        np.delete(self.labels, indices)

    @staticmethod
    def combine_datasets(datasets: List[Dataset]) -> [np.ndarray, np.ndarray]:
        data = [dataset.get_data for dataset in datasets]
        labels = [dataset.get_labels for dataset in datasets]
        return np.vstack(data), np.vstack(labels)
