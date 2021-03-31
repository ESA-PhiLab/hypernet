import abc
from copy import copy
from math import ceil
from os import PathLike
from random import shuffle
from itertools import product
from collections import Iterable

import torch
import numpy as np
from typing import List
from keras.utils import to_categorical

from python_research.io import load_data

HEIGHT = 0
WIDTH = 1
DEPTH = 2


class Dataset:

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def get_data(self) -> np.ndarray:
        """
        :return: Data from a given dataset
        """
        return self.data

    def get_labels(self) -> np.ndarray:
        """
        :return: Labels from a given dataset
        """
        return self.labels

    def get_one_hot_labels(self, classes_count: int=None):
        if classes_count is None:
            classes_count = len(np.unique(self.labels))
        return to_categorical(self.labels, classes_count)

    @property
    def min(self):
        return np.amin(self.data)

    @property
    def max(self):
        return np.amax(self.data)

    @property
    def shape(self):
        return self.data.shape

    def vstack(self, to_stack: np.ndarray):
        self.data = np.vstack([self.data, to_stack])

    def hstack(self, to_stack: np.ndarray):
        self.labels = np.hstack([self.labels, to_stack])

    def expand_dims(self, axis: int=0, inplace: bool=True):
        if inplace:
            self.data = np.expand_dims(self.data, axis=axis)
        else:
            return np.expand_dims(self.data, axis=axis)

    def normalize_min_max(self, min_: float=None, max_: float=None,
                          inplace: bool=True):
        """
        Normalize data using Min Max normalization: (data - min) / (max - min)
        :param min_: Minimal value for normalization, if not specified,
                     it will be deducted from data
        :param max_: Maximal value for normalization, if not specified,
                     it will be deducted from data
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return: If inplace is True - return None,
                 if inplace is False - return normalized (data, labels)
        """
        if min_ is None and max_ is None:
            min_ = np.amin(self.get_data())
            max_ = np.amax(self.get_data())
            if inplace:
                self.data = (self.get_data() - min_) / (max_ - min_)
            else:
                return (self.get_data() - min_) / (max_ - min_)
        elif min_ is not None and max_ is not None:
            if inplace:
                self.data = (self.get_data() - min_) / (max_ - min_)
            else:
                return(self.get_data() - min_) / (max_ - min_)

    def standardize(self, mean: float=None, std: float=None,
                    inplace: bool=True):
        """
        Standardize data using mean and std.
        :param mean: Mean value for standardization, if not specified,
                     it will be deducted from data
        :param std: Std value for standardization, if not specified,
                     it will be deducted from data
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return: If inplace is True - return None,
                 if inplace is False - return normalized (data, labels)
        """
        if mean is None and std is None:
            mean = np.mean(self.get_data())
            std = np.std(self.get_data())
        if inplace:
            self.data = (self.data - mean) / std
        else:
            return (self.data - mean) / std

    def normalize_labels(self):
        """
        Normalize label values so that they start from 0.
        :return: None
        """
        self.labels = self.labels - 1

    def delete_by_indices(self, indices: Iterable):
        """
        Delete a chunk of data given as indices
        :param indices: Indices to delete from both data and labels arrays
        :return: None
        """
        self.data = np.delete(self.data, indices, axis=HEIGHT)
        self.labels = np.delete(self.labels, indices, axis=HEIGHT)

    def convert_to_tensors(self, inplace: bool=True, device: str='cpu'):
        """
        Convert data and labels from torch tensors.
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :param device: Device on which tensors should be alocated
        :return:
        """
        if inplace:
            self.data = torch.from_numpy(self.get_data()).float().to(device)
            self.labels = torch.from_numpy(self.get_labels()).float().to(device)
        else:
            return torch.from_numpy(self.get_data()).to(device), \
                   torch.from_numpy(self.get_labels()).to(device)

    def convert_to_numpy(self, inplace: bool=True):
        """
        Convert data and labels to numpy
        :param inplace: Whether to change data in-place (True) or return
                        normalized data and labels
        :return:
        """
        if inplace:
            self.data = self.data.numpy()
            self.labels = self.labels.numpy()
        else:
            return self.data.numpy(), self.labels.numpy()

    def __len__(self) -> int:
        """
        Method providing a size of the dataaset (number of samples)
        :return: Size of the dataset
        """
        return len(self.labels)

    def __getitem__(self, item) -> [np.ndarray, np.ndarray]:
        """
        Method supporting integer indexing
        :param item: Index or Iterable of indices pointing at elements to be
                     returned
        :return: Data at given indexes
        """
        sample_x = self.data[item, ...]
        sample_y = self.labels[item]
        return sample_x, sample_y


class HyperspectralDataset(Dataset):
    """
    Class representing hyperspectral data in a form of samples prepared for
    training and classification (1D or 3D). For 1D samples, data will have
    the following dimensions: [SAMPLES_COUNT, NUMBER_OF_BANDS], where for 3D
    samples dimensions will be [SAMPLES_COUNT,
                                NEIGHBOURHOOD_SIZE,
                                NEIGHBOURHOOD_SIZE,
                                NUMBER_OF_BANDS].
    """
    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike],
                 neighbourhood_size: int = 1,
                 background_label: int = 0):
        if type(dataset) is np.ndarray and type(ground_truth) is np.ndarray:
            raw_data = dataset
            ground_truth = ground_truth
        elif type(dataset) is str and type(ground_truth) is str:
            raw_data = load_data(dataset)
            ground_truth = load_data(ground_truth)
        else:
            raise TypeError("Dataset and ground truth should be "
                            "provided either as a string or a numpy array, "
                            "not {}".format(type(dataset)))
        data, labels = self._prepare_samples(raw_data,
                                             ground_truth,
                                             neighbourhood_size,
                                             background_label)
        super(HyperspectralDataset, self).__init__(data, labels)

    @staticmethod
    def _get_padded_cube(data, padding_size):
        x = copy(data)
        v_padding = np.zeros((padding_size, x.shape[WIDTH], x.shape[DEPTH]))
        x = np.vstack((v_padding, x))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[HEIGHT], padding_size, x.shape[DEPTH]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    @staticmethod
    def _prepare_1d(raw_data: np.ndarray,
                    ground_truth: np.ndarray,
                    background_label: int):
        samples, labels = list(), list()
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(raw_data[y, x, ...])
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def _prepare_3d(self, raw_data: np.ndarray,
                    ground_truth: np.ndarray,
                    neighbourhood_size: int,
                    background_label: int):
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        padding_size = neighbourhood_size % ceil(float(neighbourhood_size) / 2.)
        padded_cube = self._get_padded_cube(raw_data, padding_size)
        samples, labels = list(), list()
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(padded_cube[y:y + padding_size * 2 + 1,
                                          x:x + padding_size * 2 + 1, ...])
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def _prepare_samples(self, raw_data: np.ndarray,
                         ground_truth: np.ndarray,
                         neighbourhood_size: int,
                         background_label: int):
        if neighbourhood_size > 1:
            samples, labels = self._prepare_3d(raw_data,
                                               ground_truth,
                                               neighbourhood_size,
                                               background_label)
        else:
            samples, labels = self._prepare_1d(raw_data,
                                               ground_truth,
                                               background_label)
        return (np.array(samples).astype(np.float64),
                np.array(labels).astype(np.uint8))


class Subset(abc.ABC):

    @abc.abstractmethod
    def extract_subset(self, *args, **kwargs) -> [np.ndarray, np.ndarray]:
        """"
        Extract some part of a given dataset
        """


class BalancedSubset(Dataset, Subset):
    """
    Extracted a subset where all classes have the same number of samples
    """

    def __init__(self, dataset: Dataset,
                 samples_per_class: int,
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset,
                                           samples_per_class,
                                           delete_extracted)
        super(BalancedSubset, self).__init__(data, labels)

    @staticmethod
    def _collect_indices_to_extract(classes: List[int],
                                    labels: np.ndarray,
                                    samples_per_class: int):
        indices_to_extract = []
        for label in classes:
            class_indices = list(np.where(labels == label)[0])
            shuffle(class_indices)
            if 0 < samples_per_class < 1:
                samples_to_extract = int(len(class_indices) * samples_per_class)
                indices_to_extract += class_indices[0:samples_to_extract]
            else:
                indices_to_extract += class_indices[0:int(samples_per_class)]
        return indices_to_extract

    def extract_subset(self, dataset: Dataset,
                       samples_per_class: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        classes, counts = np.unique(dataset.get_labels(), return_counts=True)
        if np.any(counts < samples_per_class):
            raise ValueError("Chosen number of samples per class is too big "
                             "for one of the classes")
        indices_to_extract = self._collect_indices_to_extract(classes,
                                                              dataset.get_labels(),
                                                              samples_per_class)

        data = copy(dataset.get_data()[indices_to_extract, ...])
        labels = copy(dataset.get_labels()[indices_to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(indices_to_extract)

        return data, labels


class ImbalancedSubset(Dataset, Subset):
    """
    Extract a subset where samples are drawn randomly. If total_samples_count
    is a value between 0 and 1, it is treated as a percentage of dataset
    to extract.
    """
    def __init__(self, dataset: Dataset,
                 total_samples_count: float,
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset,
                                           total_samples_count,
                                           delete_extracted)
        super(ImbalancedSubset, self).__init__(data, labels)

    def extract_subset(self, dataset: Dataset,
                       total_samples_count: int,
                       delete_extracted: bool) -> [np.ndarray, np.ndarray]:
        indices = [i for i in range(len(dataset))]
        shuffle(indices)
        if 0 < total_samples_count < 1:
            total_samples_count = int(len(dataset) * total_samples_count)
        indices_to_extract = indices[0:total_samples_count]

        data = copy(dataset.get_data()[indices_to_extract, ...])
        labels = copy(dataset.get_labels()[indices_to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(indices_to_extract)

        return data, labels


class CustomSizeSubset(Dataset, Subset):
    """
    Extract a subset where number of samples for each class is provided
    separately in a list
    """
    def __init__(self, dataset: Dataset,
                 samples_count: List[int],
                 delete_extracted: bool=True):
        data, labels = self.extract_subset(dataset, samples_count,
                                           delete_extracted)
        super(CustomSizeSubset, self).__init__(data, labels)

    def extract_subset(self, dataset: Dataset, samples_count: List[int],
                       delete_extracted: bool):
        classes = np.unique(dataset.get_labels())
        to_extract = []
        for label in classes:
            indices = np.where(dataset.get_labels() == label)[0]
            shuffle(indices)
            to_extract += list(indices[0:samples_count[label]])

        data = copy(dataset.get_data()[to_extract, ...])
        labels = copy(dataset.get_labels()[to_extract, ...])

        if delete_extracted:
            dataset.delete_by_indices(to_extract)

        return data, labels


class ConcatDataset(Dataset):
    """Dataset to concatenate multiple datasets. Useful when loading patches
    of the dataset and combining them"""

    def __init__(self, datasets: List[Dataset]):
        data, labels = self.combine_datasets(datasets)
        super(ConcatDataset, self).__init__(data, labels)

    @staticmethod
    def combine_datasets(datasets: List[Dataset]) -> [np.ndarray, np.ndarray]:
        data = [dataset.get_data() for dataset in datasets]
        labels = [dataset.get_labels() for dataset in datasets]
        return np.vstack(data), np.hstack(labels)


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


class HyperspectralCube(Dataset):
    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike] = None,
                 neighbourhood_size: int = 1,
                 device: str='cpu',
                 bands=None):
        if type(dataset) is np.ndarray and type(ground_truth) is np.ndarray:
            raw_data = dataset
            ground_truth = ground_truth
        elif type(dataset) is str and type(ground_truth) is str:
            raw_data = load_data(dataset)
            ground_truth = load_data(ground_truth)
        elif type(dataset) is str and ground_truth is None:
            raw_data = load_data(dataset)
        else:
            raise TypeError("Dataset and ground truth should be "
                            "provided either as a string or a numpy array, "
                            "not {}".format(type(dataset)))
        self.neighbourhood_size = neighbourhood_size
        self.original_2d_shape = raw_data.shape[0:2]
        self.padding_size = neighbourhood_size % ceil(float(neighbourhood_size) / 2.)
        self.indexes = self._get_indexes(raw_data.shape[HEIGHT], raw_data.shape[WIDTH])
        data = self._get_padded_cube(raw_data)
        data = data.swapaxes(1, 2).swapaxes(0, 1)
        self.device = device
        self.bands = bands
        super(HyperspectralCube, self).__init__(data, ground_truth)

    @staticmethod
    def _get_indexes(height, width):
        xx, yy = np.meshgrid(range(height), range(width))
        return [(x, y) for x, y in zip(yy.flatten(), xx.flatten())]

    def _get_padded_cube(self, data):
        v_padding = np.zeros((self.padding_size, data.shape[WIDTH], data.shape[DEPTH]))
        x = np.vstack((v_padding, data))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[HEIGHT], self.padding_size, x.shape[DEPTH]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        if type(item) is list:
            batch = torch.zeros([len(item),
                                 1,
                                 self.bands,
                                 self.neighbourhood_size,
                                 self.neighbourhood_size], device=self.device)
            for sample, sample_index in enumerate(item):
                x, y = self.indexes[sample_index]
                batch[sample, 0] = self.data[:self.bands, y:y + self.padding_size * 2 + 1,
                                                x:x + self.padding_size * 2 + 1]
            return batch
        else:
            x, y = self.indexes[item]
            return self.data[:self.bands, y:y + self.padding_size * 2 + 1,
                                x:x + self.padding_size * 2 + 1]


class DataLoaderShuffle:
    def __init__(self, dataset: Dataset, batch_size: int=64):
        self.batch_size = batch_size
        self.data = dataset
        self.samples_count = len(dataset)
        self.indexes = self._get_indexes()
        self.samples_returned = 0

    def __iter__(self):
        self.samples_returned = 0
        return self

    def __next__(self):
        if (self.samples_returned + self.batch_size) > self.samples_count:
            raise StopIteration
        else:
            indexes = self.indexes[self.samples_returned:
                                   self.samples_returned + self.batch_size]
            batch = self.data[indexes]
            self.samples_returned += self.batch_size
            return batch

    def __len__(self):
        return int(self.samples_count / self.batch_size)

    def cube_2d_shape(self):
        return self.data.original_2d_shape

    def shuffle(self):
        shuffle(self.indexes)

    def sort(self):
        self.indexes = self._get_indexes()

    def _get_indexes(self):
        indexes = [x for x in range(self.samples_count)]
        return indexes