import numpy as np
from torch.utils.data import Dataset
from random import shuffle

BACKGROUND_LABEL = 0


class HyperspectralDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 ground_truth_path: str,
                 transform=None,
                 normalize: bool=True,
                 samples_per_class: int=None):
        self.x, self.y = self._transform_data(data_path, ground_truth_path)
        self.transform = transform
        self.classes = np.unique(self.y)
        if normalize:
            self.x = self._normalize()
        self.y = self.y - 1
        if samples_per_class is not None:
            self.x, self.y = self._balance_set(samples_per_class)

    @staticmethod
    def _transform_data(data_path: str, ground_truth_path: str):
        x = np.load(data_path)
        y = np.load(ground_truth_path)
        transformed = []
        labels = []
        for i, row in enumerate(x):
            for j, pixel in enumerate(row):
                if y[i, j] != BACKGROUND_LABEL:
                    sample = x[i, j, :]
                    transformed.append(sample)
                    labels.append(y[i, j])
        return np.array(transformed).astype(np.float64), np.array(labels)

    def _normalize(self):
        return (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))

    def _balance_set(self, samples_per_class: int):
        labels, count = np.unique(self.y, return_counts=True)
        if samples_per_class == 0:
            samples_per_class = min(count)
        to_remove = []
        for label in labels:
            label_indices = np.where(self.y == label)[0]
            shuffle(label_indices)
            to_remove += list(label_indices[samples_per_class:])
        x = np.delete(self.x, to_remove, axis=0)
        y = np.delete(self.y, to_remove, axis=0)
        indexes = list(range(len(y)))
        indexes.sort(key=y.__getitem__)
        soted = map(y.__getitem__, indexes)
        sorted_x = map(x.__getitem__, indexes)
        soted = list(soted)
        sorted_x = list(sorted_x)
        sorted_x = np.array(sorted_x)
        return sorted_x, np.array(soted)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        sample_x = self.x[item, ...]
        sample_y = self.y[item, ...]
        if self.transform:
            sample_x = self.transform(sample_x)
        return sample_x, sample_y
