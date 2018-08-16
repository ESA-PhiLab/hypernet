import random
from copy import copy
from math import ceil
from typing import Tuple, List

import numpy as np
from keras.utils import to_categorical
from scipy.ndimage.filters import gaussian_filter

from python_research.experiments.multiple_feature_learning.utils.data_types \
    import TrainTestIndices
from python_research.segmentation import Point

PROBABILITY_THRESHOLD = 0.5
BACKGROUND_LABEL = 0
SIGMA = 2


class Dataset:
    def __init__(self, dataset_file_path: str, gt_filepath: str,
                 no_train_samples: float, neighbours_size: Tuple[int, int],
                 validation_set_portion=0.1,
                 normalize=True,
                 classes_count: int=None,
                 train_test_indices: TrainTestIndices=None):
        self.x = np.load(dataset_file_path)
        self.y = np.load(gt_filepath)
        self.min_value = np.min(self.x)
        self.max_value = np.max(self.x)
        if normalize:
            self.x = self._normalize_data()
        self.no_train_samples = self._get_train_samples_per_class_count(no_train_samples)
        if train_test_indices is None:
            self.train_indices, self.test_indices = self._get_train_test_indices()
            self.val_indices = self._get_val_indices(validation_set_portion)
        else:
            self.train_indices = train_test_indices.train_indices
            self.test_indices = train_test_indices.test_indices
            self.val_indices = train_test_indices.val_indices

        # self._label_augmentation()

        padding_size = neighbours_size[0] % ceil(float(neighbours_size[0]) / 2.)
        padded_cube = self.get_padded_cube(padding_size)

        self.x_train, self.y_train = self._construct_sets(padded_cube, self.train_indices, padding_size)
        self.x_test, self.y_test = self._construct_sets(padded_cube, self.test_indices, padding_size)
        self.x_val, self.y_val = self._construct_sets(padded_cube, self.val_indices, padding_size)

        if classes_count is None:
            classes_count = len(np.unique(self.y)) - 1
        self.y_train = to_categorical(self.y_train - 1, classes_count)
        self.y_test = to_categorical(self.y_test - 1, classes_count)
        self.y_val = to_categorical(self.y_val - 1, classes_count)

    def __add__(self, other):
        if self.x_train.size != 0 and other.x_train.size != 0:
            self.x_train = np.concatenate([self.x_train, other.x_train], axis=0)
            self.y_train = np.concatenate([self.y_train, other.y_train], axis=0)
        if self.x_test.size != 0 and other.x_test.size != 0:
            self.x_test = np.concatenate([self.x_test, other.x_test], axis=0)
            self.y_test = np.concatenate([self.y_test, other.y_test], axis=0)
        if self.x_val.size != 0 and other.x_val.size != 0:
            self.x_val = np.concatenate([self.x_val, other.x_val], axis=0)
            self.y_val = np.concatenate([self.y_val, other.y_val], axis=0)
        self.min_value = self.min_value if self.min_value < other.min_value else other.min_value
        self.max_value = self.max_value if self.max_value > other.max_value else other.max_value
        return self

    def _normalize_data(self):
        min_ = np.min(self.x, keepdims=True)
        max_ = np.max(self.x, keepdims=True)
        return (self.x.astype(np.float64) - min_) / (max_ - min_)

    def normalize_train_test_data(self):
        self.x_train[self.x_train != 0] = (self.x_train[self.x_train != 0] - self.min_value) / (self.max_value - self.min_value)
        self.x_val[self.x_val != 0] = (self.x_val[self.x_val != 0] - self.min_value) / (self.max_value - self.min_value)

    def _get_train_samples_per_class_count(self, no_train_samples):
        labels, classes_count = np.unique(self.y, return_counts=True)
        if BACKGROUND_LABEL in labels:
            labels = np.delete(labels, BACKGROUND_LABEL)
            classes_count = np.delete(classes_count, BACKGROUND_LABEL)
        train_samples_per_class = dict.fromkeys(labels)

        # treat as a percentage of samples
        if 1 > no_train_samples > 0:
            for index, label in enumerate(sorted(train_samples_per_class.keys())):
                train_samples_per_class[label] = int(classes_count[index] * no_train_samples)

        # samples count strictly for Indiana
        elif no_train_samples == 1:
            counts = [30, 250, 250, 150, 250, 250, 20, 250, 15, 250, 250, 250,
                      150, 250, 50, 50]
            for index, label in enumerate(sorted(train_samples_per_class.keys())):
                train_samples_per_class[label] = counts[index]

        # all samples as train samples
        elif no_train_samples == -1:
            for index, label in enumerate(sorted(train_samples_per_class.keys())):
                train_samples_per_class[label] = classes_count[index]

        else:
            for index, label in enumerate(sorted(train_samples_per_class.keys())):
                train_samples_per_class[label] = int(no_train_samples)
        return train_samples_per_class

    def _get_train_test_indices(self):
        label_indices = dict()
        for x, row in enumerate(self.y):
            for y, label in enumerate(row):
                if label != BACKGROUND_LABEL:
                    if label not in label_indices.keys():
                        label_indices[label] = [Point(x, y)]
                    else:
                        label_indices[label].append(Point(x, y))
        return self._randomly_select_samples(label_indices)

    def _get_val_indices(self, validation_set_portion):
        val_samples_per_label = dict.fromkeys(self.no_train_samples.keys())
        for label in self.no_train_samples.keys():
            val_samples_per_label[label] = int(self.no_train_samples[label] * validation_set_portion)
        val_indices = dict()
        for label in self.train_indices.keys():
            val_indices[label] = self.train_indices[label][0:val_samples_per_label[label]]
            self.train_indices[label] = self.train_indices[label][val_samples_per_label[label]:]
        return val_indices

    def _randomly_select_samples(self, label_indices):
        train_indices = dict()
        test_indices = dict()
        for label in sorted(label_indices):
            random.shuffle(label_indices[label])
            train_indices[label] = label_indices[label][
                                   0:self.no_train_samples[label]]
            test_indices[label] = label_indices[label][
                                  self.no_train_samples[label]:]
        return train_indices, test_indices

    def get_padded_cube(self, padding_size: int):
        x = copy(self.x)
        v_padding = np.zeros((padding_size, x.shape[1], x.shape[2]))
        x = np.vstack((v_padding, x))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[0], padding_size, x.shape[2]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    @staticmethod
    def _construct_sets(data_cube, indices, padding_size):
        x, y = list(), list()
        for label in indices:
            for point in indices[label]:
                x.append(
                    copy(data_cube[point.x:point.x + padding_size * 2 + 1,
                         point.y:point.y + padding_size * 2 + 1, :]))
                y.append(label)
        return np.array(x), np.array(y)

    def _construct_1d_sets(self, indices):
        x, y = list(), list()
        for label in indices:
            for point in indices[label]:
                sample = copy(self.x[point.x, point.y, :])
                sample = sample.reshape(sample.shape[-1], 1)
                x.append(sample)
                y.append(label)
        return np.array(x).astype(np.float64), np.array(y).astype(np.float64)

    def _get_neighbours(self, point: Point) -> List[Point]:
        neighbours = []
        adjacency = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if
                     not (i == j == 0)]
        for dx, dy in adjacency:
            if 0 <= point.x + dx < self.x.shape[1] and 0 <= point.y + dy < \
                    self.x.shape[0]:
                x = point.x + dx
                y = point.y + dy
                neighbours.append(Point(x, y))
        return neighbours

    def _label_augmentation(self):
        min_labels_count = min(np.delete(self.no_train_samples, BACKGROUND_LABEL))
        max_labels_count = max(np.delete(self.no_train_samples, BACKGROUND_LABEL))
        label_probabilities = [1 - (label_count - min_labels_count) /
                               (max_labels_count - min_labels_count)
                               for label_count in self.no_train_samples]
        neighbours = set()
        for label in self.train_indices.keys():
            for point in self.train_indices[label]:
                if label_probabilities[label] > PROBABILITY_THRESHOLD:
                    neighbours.update(self._get_neighbours(point))
            neighbours.update(self.train_indices[label])
            self.train_indices[label] = list(neighbours)
            neighbours = set()

    def _apply_gaussian_filter(self):
        filtered_cube = np.empty(self.x.shape)
        for wavelength in range(self.x.shape[-1]):
            filtered_cube[:, :, wavelength] = gaussian_filter(self.x[:, :, wavelength], SIGMA)
        return filtered_cube
