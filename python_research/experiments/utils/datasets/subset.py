import abc
import numpy as np
from copy import copy
from random import shuffle
from typing import List, Iterable
from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset


class Subset(abc.ABC):

    @abc.abstractmethod
    def extract_subset(self, *args, **kwargs) -> [np.ndarray, np.ndarray]:
        """"
        Extract some part of a given dataset
        """


class BalancedSubset(Dataset, Subset):

    def __init__(self, dataset: Dataset,
                 samples_per_class: int,
                 delete_extracted: bool=True):
        self.data, self.labels = self.extract_subset(dataset,
                                                     samples_per_class,
                                                     delete_extracted)

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
    def _collect_indices_to_extract(classes: List[int],
                                    labels: np.ndarray,
                                    samples_per_class: int):
        indices_to_extract = []
        for label in classes:
            class_indices = list(np.where(labels == label)[0])
            shuffle(class_indices)
            indices_to_extract += class_indices[0:samples_per_class]
        return indices_to_extract

    def extract_subset(self, dataset: Dataset,
                       samples_per_class: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        classes, counts = np.unique(dataset.get_labels, return_counts=True)
        if np.any(counts < samples_per_class):
            raise ValueError("Chosen number of samples per class is too big "
                             "for one of the classes")
        indices_to_extract = self._collect_indices_to_extract(classes,
                                                              dataset.get_labels,
                                                              samples_per_class)

        data = copy(dataset.get_data[indices_to_extract, ...])
        labels = copy(dataset.get_data[indices_to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(indices_to_extract)

        return data, labels


class UnbalancedSubset(Dataset, Subset):

    def __init__(self, dataset: Dataset,
                 total_samples_count: int,
                 delete_extracted: bool=True):
        self.data, self.labels = self.extract_subset(dataset,
                                                     total_samples_count,
                                                     delete_extracted)

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

    def extract_subset(self, dataset: Dataset,
                       total_samples_count: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        indices = [i for i in range(len(dataset))]
        shuffle(indices)

        indices_to_extract = indices[0:total_samples_count]

        data = copy(dataset.get_data[indices_to_extract, ...])
        labels = copy(dataset.get_data[indices_to_extract, ...])

        if delete_extracted:
            np.delete(dataset.get_data, indices_to_extract)
            np.delete(dataset.get_data, indices_to_extract)

        return data, labels

