import numpy as np
from random import shuffle
from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset

BACKGROUND_LABEL = 0


class OrderedDataLoader:
    """
    Shuffling is performed only withing classes, the order of the
    returned classes is fixed.
    """
    def __init__(self, dataset: Dataset, batch_size: int=64,
                 use_tensors: bool=True):
        self.batch_size = batch_size
        self.label_samples_indices = self._get_label_samples_indices(dataset)
        self.samples_returned = 0
        self.samples_count = len(dataset)
        self.indexes = self._get_indexes()
        self.data = dataset
        if use_tensors:
            self.data.convert_to_tensors()

    def __iter__(self):
        self.indexes = self._get_indexes()
        return self

    def __next__(self):
        if (self.samples_returned + self.batch_size) > self.samples_count:
            self.samples_returned = 0
            raise StopIteration
        else:
            indexes = self.indexes[self.samples_returned:
                                   self.samples_returned + self.batch_size]
            batch = self.data[indexes]
            self.samples_returned += self.batch_size
            return batch

    def _get_indexes(self):
        labels = self.label_samples_indices.keys()
        indexes = []
        for label in labels:
            shuffle(self.label_samples_indices[label])
            indexes += self.label_samples_indices[label]
        return indexes

    @staticmethod
    def _get_label_samples_indices(dataset):
        labels = np.unique(dataset.get_labels())
        label_samples_indices = dict.fromkeys(labels)
        for label in label_samples_indices:
            label_samples_indices[label] = list(np.where(dataset.get_labels() == label)[0])
        return label_samples_indices

