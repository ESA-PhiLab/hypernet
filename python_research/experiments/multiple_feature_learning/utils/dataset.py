import random
import numpy as np
from math import ceil
from typing import Tuple, List
from copy import copy, deepcopy
from keras.utils import to_categorical
from python_research.segmentation import Point
from python_research.experiments.multiple_feature_learning.utils.data_types \
    import TrainTestIndices


PROBABILITY_THRESHOLD = 0.5
BACKGROUND_LABEL = 0


class Dataset:
    def __init__(self, dataset_file_path: str, gt_filepath: str,
                 no_train_samples: float, neighbours_size: Tuple[int, int],
                 validation_set_portion=0.1,
                 train_test_indices: TrainTestIndices=None):
        self.x = np.load(dataset_file_path)
        self.y = np.load(gt_filepath)
        self.x = self._normalize_data()
        self.no_train_samples = self._get_train_samples_per_class_count(no_train_samples)
        if train_test_indices is None:
            self.train_indices, self.test_indices = self._get_train_test_indices()
        else:
            self.train_indices = train_test_indices.train_indices
            self.test_indices = train_test_indices.test_indices
        self.val_indices = self._get_val_indices(validation_set_portion)

        self._label_augmentation()

        padding_size = neighbours_size[0] % ceil(float(neighbours_size[0]) / 2.)
        padded_cube = self.get_padded_cube(padding_size)

        self.x_train, self.y_train = self._construct_sets(padded_cube, self.train_indices, padding_size)
        self.x_test, self.y_test = self._construct_sets(padded_cube, self.test_indices, padding_size)
        self.x_val, self.y_val = self._construct_sets(padded_cube, self.val_indices, padding_size)

        labels_count = len(np.unique(self.y)) - 1
        self.y_train = to_categorical(self.y_train - 1, labels_count)
        self.y_test = to_categorical(self.y_test - 1, labels_count)
        self.y_val = to_categorical(self.y_val - 1, labels_count)

    def _normalize_data(self):
        min_ = np.min(self.x, keepdims=True)
        max_ = np.max(self.x, keepdims=True)
        return (self.x - min_) / (max_ - min_)

    def _get_train_samples_per_class_count(self, no_train_samples):
        _, classes_count = np.unique(self.y, return_counts=True)

        # treat as a percentage of samples
        if 1 > no_train_samples > 0:
            train_samples_per_class = np.array(classes_count * no_train_samples,
                                               dtype=np.uint16)

        # samples count strictly for Indiana
        elif no_train_samples == 1:
            train_samples_per_class = [0, 30, 250, 250, 150, 250, 250, 20, 250,
                                       15, 250, 250, 250, 150, 250, 50, 50]
            train_samples_per_class = np.array(train_samples_per_class)
        else:
            train_samples_per_class = [no_train_samples for _ in classes_count]
            train_samples_per_class = np.array(train_samples_per_class)
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
        val_samples_per_class = self.no_train_samples * validation_set_portion
        val_samples_per_class = np.array(val_samples_per_class, dtype=np.int16)
        val_indices = dict()
        for label in self.train_indices.keys():
            val_indices[label] = self.train_indices[label][0:val_samples_per_class[label]]
            self.train_indices[label] = self.train_indices[label][val_samples_per_class[label]:]
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

    def _construct_sets(self, data_cube, indices, padding_size):
        x, y = list(), list()
        for label in indices:
            for point in indices[label]:
                x.append(
                    copy(data_cube[point.x:point.x + padding_size * 2 + 1,
                         point.y:point.y + padding_size * 2 + 1, :]))
                y.append(label)
        return np.array(x), np.array(y)

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
        train_indices = deepcopy(self.train_indices)
        for label in self.train_indices.keys():
            for point in self.train_indices[label]:
                if label_probabilities[label] > PROBABILITY_THRESHOLD:
                    neighbours = self._get_neighbours(point)
                    train_indices[label] += neighbours
        self.train_indices = train_indices

import cProfile
import re
import pstats
def run():
    d = Dataset("C:\\Users\MMyller\Documents\datasets\Indian\Indian_pines_corrected.npy",
                "C:\\Users\MMyller\Documents\datasets\Indian\Indian_pines_gt.npy", 1, (5,5))
    a = 5
cProfile.run('run()', 'elo')
# p = pstats.Stats('elo')
# p.strip_dirs().sort_stats(-1).print_stats()