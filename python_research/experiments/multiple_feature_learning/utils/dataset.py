import random
import numpy as np
from math import ceil
from typing import Tuple
from copy import copy
from keras.utils import to_categorical
from python_research.experiments.multiple_feature_learning.utils.data_types import (
    TrainTestIndices
)


class Dataset:
    def __init__(
        self,
        dataset_file_path: str,
        gt_filepath: str,
        no_train_samples: float,
        neighbours_size: Tuple[int, int],
        validation_set_portion=0.1,
        train_test_indices: TrainTestIndices=None
    ):
        if no_train_samples < 10:
            raise ValueError('Number of training samples must be greater or equal to 10')
        self.x = np.load(dataset_file_path)
        self.y = np.load(gt_filepath)
        self.neighbours_size = neighbours_size
        self.labels, self.classes_count = np.unique(self.y, return_counts=True)
        self.x = self._normalize_data()
        self.no_train_samples = self._get_train_samples_per_class_count(no_train_samples)
        if train_test_indices is None:
            self.train_indices, self.test_indices = self._get_train_test_indices()
        else:
            self.train_indices = train_test_indices.train_indices
            self.test_indices = train_test_indices.test_indices
        self.x_train, self.x_test, self.y_train, self.y_test = self._train_test_split()
        self.x_train, self.y_train, self.x_val, self.y_val = self._train_val_split(
            validation_set_portion)
        self.y_train = to_categorical(self.y_train - 1, len(self.labels) - 1)
        self.y_test = to_categorical(self.y_test - 1, len(self.labels) - 1)
        self.y_val = to_categorical(self.y_val - 1, len(self.labels) - 1)

    def _normalize_data(self):
        min_ = np.min(self.x, keepdims=True)
        max_ = np.max(self.x, keepdims=True)
        return (self.x - min_) / (max_ - min_)

    def _get_train_samples_per_class_count(self, no_train_samples):
        # treat as a percentage of samples
        if 1 > no_train_samples > 0:
            train_samples_per_class = self.classes_count * no_train_samples
            return np.array(train_samples_per_class, dtype=np.int16)
        # samples count strictly for Indiana
        elif no_train_samples == 1:
            train_samples_per_class = [
                0, 30, 250, 250, 150, 250, 250, 20, 250, 15, 250, 250, 250, 150, 250, 50, 50
            ]
            return np.array(train_samples_per_class)
        else:
            train_samples_per_class = [no_train_samples for _ in self.classes_count]
            return np.array(train_samples_per_class)

    def _get_train_test_indices(self):
        label_indices = dict()
        for i, row in enumerate(self.y):
            for j, label in enumerate(row):
                if label != 0:
                    if label not in label_indices.keys():
                        label_indices[label] = [(i, j)]
                    else:
                        label_indices[label].append((i, j))
        return self._randomly_select_samples(label_indices)

    def _randomly_select_samples(self, label_indices):
        train_indices = dict()
        test_indices = dict()
        for label in sorted(label_indices):
            random.shuffle(label_indices[label])
            train_indices[label] = label_indices[label][0:self.no_train_samples[label]]
            test_indices[label] = label_indices[label][self.no_train_samples[label]:]
        return train_indices, test_indices

    def _add_padding(self, padding_size: int):
        x = copy(self.x)
        v_padding = np.zeros((padding_size, x.shape[1], x.shape[2]))
        x = np.vstack((v_padding, x))
        x = np.vstack((x, v_padding))
        h_padding = np.zeros((x.shape[0], padding_size, x.shape[2]))
        x = np.hstack((h_padding, x))
        x = np.hstack((x, h_padding))
        return x

    def _construct_sets(self, x, padding_size):
        x_train, x_test, y_train, y_test = list(), list(), list(), list()
        for label in self.train_indices:
            for coords in self.train_indices[label]:
                x_train.append(
                    copy(x[
                        coords[0]:coords[0] + padding_size * 2 + 1,
                        coords[1]:coords[1] + padding_size * 2 + 1,
                        :
                    ]))
                y_train.append(label)
        for label in self.test_indices:
            for coords in self.test_indices[label]:
                x_test.append(copy(x[
                    coords[0]:coords[0] + padding_size * 2 + 1,
                    coords[1]:coords[1] + padding_size * 2 + 1,
                    :
                ]))
                y_test.append(label)
        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def _train_test_split(self):
        padding_size = self.neighbours_size[0] % ceil(float(self.neighbours_size[0]) / 2.)
        x = self._add_padding(padding_size)
        x_train, x_test, y_train, y_test = self._construct_sets(x, padding_size)
        return x_train, x_test, y_train, y_test

    def _train_val_split(self, validation_set_portion):
        labels, classes_count = np.unique(self.y_train, return_counts=True)
        val_samples_per_class = classes_count * validation_set_portion
        val_samples_per_class = np.array(val_samples_per_class, dtype=np.int16)
        x_train, y_train, x_val, y_val = list(), list(), list(), list()
        for label in labels:
            label_samples = self.x_train[self.y_train == label]
            train_label_samples = label_samples[0:val_samples_per_class[label - 1]]
            val_label_samples = label_samples[val_samples_per_class[label - 1]:]
            for sample in train_label_samples:
                x_val.append(copy(sample))
                y_val.append(label)
            for sample in val_label_samples:
                x_train.append(copy(sample))
                y_train.append(label)
        return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)
