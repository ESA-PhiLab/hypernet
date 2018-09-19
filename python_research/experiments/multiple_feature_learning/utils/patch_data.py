from math import ceil
import numpy as np
from itertools import product
from typing import Tuple
from copy import copy
from random import shuffle
from scipy.ndimage.filters import gaussian_filter
from keras.utils import to_categorical

BACKGROUND_LABEL = 0


class PatchData:
    def __init__(self, dataset: [str, np.ndarray], ground_truth: [str, np.ndarray],
                 neighbourhood_size: Tuple[int, int], sigmas=None):
        if type(dataset) is str and type(ground_truth) is str:
            x = np.load(dataset)
            y = np.load(ground_truth)
        elif type(dataset) is np.ndarray and type(ground_truth) is np.ndarray:
            x = dataset
            y = ground_truth
        else:
            raise TypeError("The 'dataset' and 'ground_truth' arguments have to be a "
                            "string (a path to dataset in .npy format) or a numpy array itself")
        if not all(np.array(neighbourhood_size) == 1):
            self.data, self.labels = self._get_samples(x, y, neighbourhood_size)
        else:
            self.data, self.labels = self.get_samples_1d(x, y)
        self.min = None
        self.max = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.val_indices = None

    def __iadd__(self, other):
        self.data = np.concatenate([self.data, other.data], axis=0)
        self.labels = np.concatenate([self.labels, other.labels], axis=0)
        return self

    def _get_samples(self, raw_data, ground_truth, neighbourhood_size):
        col_indexes = [x for x in range(0, raw_data.shape[1])]
        row_indexes = [y for y in range(0, raw_data.shape[0])]
        padding_size = neighbourhood_size[0] % ceil(float(neighbourhood_size[0]) / 2.)
        padded_cube = self.get_padded_cube(raw_data, padding_size)
        samples, labels = list(), list()
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != BACKGROUND_LABEL:
                samples.append(copy(padded_cube[y:y + padding_size * 2 + 1,
                         x:x + padding_size * 2 + 1, :]))
                labels.append(ground_truth[y, x])
        return np.array(samples).astype(np.float64), np.array(labels).astype(np.uint16)

    @staticmethod
    def get_samples_1d(raw_data, ground_truth):
        col_indexes = [x for x in range(0, raw_data.shape[1])]
        row_indexes = [y for y in range(0, raw_data.shape[0])]
        samples, labels = list(), list()
        for x, y in product(col_indexes, row_indexes):
            if ground_truth[y, x] != BACKGROUND_LABEL:
                sample = copy(raw_data[y, x, :])
                sample = sample.reshape(sample.shape[-1], 1)
                samples.append(sample)
                labels.append(ground_truth[y, x])
        return np.array(samples).astype(np.float64), np.array(labels).astype(np.uint16)

    @staticmethod
    def get_padded_cube(data, padding_size: int):
        x = copy(data)
        v_padding = np.zeros((padding_size, x.shape[1], x.shape[2]))
        x = np.vstack((v_padding, x))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[0], padding_size, x.shape[2]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    def _get_val_indices(self, val_set_portion):
        labels, count = np.unique(self.labels, return_counts=True)
        label_indices = dict.fromkeys(labels)
        for label in labels:
            label_indices[label] = np.where(self.labels == label)[0]
            samples = len(label_indices[label])
            shuffle(label_indices[label])
            label_indices[label] = label_indices[label][0:int(samples * val_set_portion)]
        return label_indices

    def train_val_split(self, val_set_portion: float=0.1, val_indices=None):
        if val_indices is None:
            val_indices = self._get_val_indices(val_set_portion)
            indices = []
            for label in val_indices.keys():
                indices = np.append(indices, val_indices[label]).astype(np.uint32)
            self.val_indices = indices
        else:
            self.val_indices = val_indices
        self.x_val = copy(self.data[self.val_indices, ...])
        self.y_val = copy(self.labels[self.val_indices])
        self.x_train = np.delete(self.data, self.val_indices, axis=0)
        self.y_train = np.delete(self.labels, self.val_indices, axis=0)

    def add_gaussian_augmented(self, sigma):
        label_indices = dict.fromkeys(np.unique(self.labels))
        most_numerous_label = 0
        for label in label_indices:
            label_indices[label] = np.where(self.labels == label)[0]
            if len(label_indices[label]) > most_numerous_label:
                most_numerous_label = len(label_indices[label])
        augmented = []
        new_labels = []
        for label in label_indices:
            if len(label_indices[label]) * 2 > most_numerous_label:
                to_add = most_numerous_label - len(label_indices[label])
            else:
                to_add = len(label_indices[label])
            shuffle(label_indices[label])
            for i in range(to_add):
                a = gaussian_filter(self.data[label_indices[label][i], ...], sigma=sigma)
                augmented.append(a)
                new_labels.append(label)
        self.x_train = np.concatenate([self.x_train, augmented], axis=0)
        self.y_train = np.concatenate([self.y_train, new_labels], axis=0)

    def normalize_data(self, classes_count):
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self.x_train = (self.x_train - self.min) / (self.max - self.min)
        self.x_val = (self.x_val - self.min) / (self.max - self.min)
        self.y_train = to_categorical(self.y_train - 1, classes_count)
        self.y_val = to_categorical(self.y_val - 1, classes_count)