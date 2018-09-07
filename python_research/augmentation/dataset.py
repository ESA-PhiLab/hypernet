import torch
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
        return torch.from_numpy(sample_x), torch.from_numpy(sample_y)


class CustomDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int=64):
        self.data = dataset
        self.batch_size = batch_size
        self.label_samples_indices = self._get_label_samples_indices(dataset)
        self.samples_returned = 0
        self.samples_count = len(dataset)
        self.indexes = self._get_indexes()

    def __iter__(self):
        self.indexes = self._get_indexes()
        return self

    def __next__(self):
        if (self.samples_returned + self.batch_size) > self.samples_count:
            self.samples_returned = 0
            raise StopIteration
        else:
            indexes = self.indexes[self.samples_returned:self.samples_returned + self.batch_size]
            batch = self.data[indexes]
            self.samples_returned += self.batch_size
            return batch

    def _get_indexes(self):
        labels = list(range(len(self.label_samples_indices)))
        shuffle(labels)
        indexes = []
        for label in labels:
            shuffle(self.label_samples_indices[label])
            indexes += self.label_samples_indices[label]
        return indexes

    @staticmethod
    def _get_label_samples_indices(dataset):
        labels = np.unique(dataset.y)
        label_samples_indices = dict.fromkeys(labels)
        for label in label_samples_indices:
            label_samples_indices[label] = list(np.where(dataset.y == label)[0])
        return label_samples_indices

