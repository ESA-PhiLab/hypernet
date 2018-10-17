import abc
from math import ceil
from copy import copy
from os import PathLike
from itertools import product

import numpy as np

from python_research.experiments.utils.io import load_data

HEIGHT = 0
WIDTH = 1
DEPTH = 2


class Dataset(abc.ABC):
    """Interface for Hyperspectral Dataset"""
    pass


class HyperspectralDataset(Dataset):

    def __init__(self, dataset: [np.ndarray, PathLike],
                 ground_truth: [np.ndarray, PathLike],
                 neighbourhood_size: int=1,
                 background_label: int=0):
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
        self.data, self.labels = self.prepare_samples(raw_data,
                                                      ground_truth,
                                                      neighbourhood_size,
                                                      background_label)

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
    def prepare_1d(raw_data: np.ndarray,
                   ground_truth: np.ndarray,
                   background_label: int):
        samples, labels = list(), list()
        col_indexes = [x for x in range(0, raw_data.shape[WIDTH])]
        row_indexes = [y for y in range(0, raw_data.shape[HEIGHT])]
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != background_label:
                sample = copy(raw_data[y, x, ...])
                sample = sample.reshape(sample.shape[-1], 1)
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return samples, labels

    def prepare_3d(self, raw_data: np.ndarray,
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

    def prepare_samples(self, raw_data: np.ndarray,
                        ground_truth: np.ndarray,
                        neighbourhood_size: int,
                        background_label: int):
        if neighbourhood_size > 1:
            samples, labels = self.prepare_3d(raw_data,
                                              ground_truth,
                                              neighbourhood_size,
                                              background_label)
        else:
            samples, labels = self.prepare_1d(raw_data,
                                              ground_truth,
                                              background_label)
        return (np.array(samples).astype(np.float32),
                np.array(labels).astype(np.int8))



