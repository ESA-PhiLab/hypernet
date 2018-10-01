import numpy as np
from copy import copy
from random import shuffle
from collections import OrderedDict
from keras.utils import to_categorical

BACKGROUND_LABEL = 0


class UnbalancedData:
    def __init__(self, file_path, gt_path, samples_number):
        self.x = np.load(file_path)
        self.y = np.load(gt_path)
        self.x_train = None
        self.y_train = None
        self.x_val = []
        self.y_val = []
        self.x_test = None
        self.y_test = None
        self.construct_train_val_sets(samples_number)
        self.normalize_sets()

    def reshape_data(self):
        data = []
        labels = []
        for i, row in enumerate(self.x):
            for j, pixel in enumerate(row):
                if self.y[i, j] != BACKGROUND_LABEL:
                    sample = copy(self.x[i, j, :])
                    data.append(sample.reshape((sample.shape[-1], 1)))
                    labels.append(self.y[i, j])
        return np.array(data), np.array(labels)

    def construct_train_val_sets(self, samples_number):
        data, labels = self.reshape_data()
        samples_count = len(data)
        indexes = [index for index in range(samples_count)]
        shuffle(indexes)
        train_indexes = indexes[:samples_number]
        test_indexes = indexes[samples_number:]
        self.x_train = data[train_indexes]
        self.y_train = labels[train_indexes]
        val_indexes = dict.fromkeys(np.unique(self.y_train))
        indexes = []
        for label in val_indexes:
            label_indexes = np.where(self.y_train == label)[0]
            label_indexes = list(label_indexes[:int(len(label_indexes) * 0.1)])
            indexes += label_indexes
        self.x_val = self.x_train[indexes, ...]
        self.y_val = self.y_train[indexes]
        self.x_train = np.delete(self.x_train, indexes, axis=0)
        self.y_train = np.delete(self.y_train, indexes, axis=0)
        self.x_test = data[test_indexes]
        self.y_test = labels[test_indexes]

        train_labels = np.concatenate((self.y_train, self.y_val), axis=0)
        self.counts = OrderedDict.fromkeys(np.delete(np.unique(self.y), BACKGROUND_LABEL), 0)
        for sample in train_labels:
            self.counts[sample] += 1

    def normalize_sets(self):
        min_ = np.min(self.x_train) if np.min(self.x_train) < np.min(self.x_val) else np.min(self.x_val)
        max_ = np.max(self.x_train) if np.max(self.x_train) > np.max(self.x_val) else np.max(self.x_val)
        self.x_train = (self.x_train.astype(np.float64) - min_) / (max_ - min_)
        self.x_val = (self.x_val.astype(np.float64) - min_) / (max_ - min_)
        self.x_test = (self.x_test.astype(np.float64) - min_) / (max_ - min_)
        self.y_train = to_categorical(self.y_train - 1, len(np.unique(self.y)) - 1)
        self.y_val = to_categorical(self.y_val - 1, len(np.unique(self.y)) - 1)
        self.y_test = to_categorical(self.y_test - 1, len(np.unique(self.y)) - 1)