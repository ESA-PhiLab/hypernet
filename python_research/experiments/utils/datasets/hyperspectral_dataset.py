import torch
from collections import Iterable
from math import ceil
from copy import copy
from os import PathLike
from itertools import product
from keras.utils import to_categorical

import numpy as np

from python_research.experiments.utils.io import load_data

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

    def convert_to_tensors(self, inplace: bool=True):
        if inplace:
            self.data = torch.from_numpy(self.get_data())
            self.labels = torch.from_numpy(self.get_labels())
        else:
            return torch.from_numpy(self.get_data()), \
                   torch.from_numpy(self.get_labels())

    def convert_to_numpy(self, inplace: bool=True):
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
    training and classification (1D or 3D)
    """
    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike],
                 neighborhood_size: int = 1,
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
        data, labels = self.prepare_samples(raw_data,
                                            ground_truth,
                                            neighborhood_size,
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
                    neighborhood_size: int,
                    background_label: int):
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        padding_size = neighborhood_size % ceil(float(neighborhood_size) / 2.)
        padded_cube = self._get_padded_cube(raw_data, padding_size)
        samples, labels = list(), list()
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(padded_cube[y:y + padding_size * 2 + 1,
                                          x:x + padding_size * 2 + 1, ...])
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def prepare_samples(self, raw_data: np.ndarray,
                        ground_truth: np.ndarray,
                        neighborhood_size: int,
                        background_label: int):
        if neighborhood_size > 1:
            samples, labels = self._prepare_3d(raw_data,
                                               ground_truth,
                                               neighborhood_size,
                                               background_label)
        else:
            samples, labels = self._prepare_1d(raw_data,
                                               ground_truth,
                                               background_label)
        return (np.array(samples).astype(np.float64),
                np.array(labels).astype(np.uint8))
