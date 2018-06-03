import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.utils import to_categorical

from experiments.EUROMICRO2018.utils.load_data import *


class Dataset:
    def __init__(self, name: str, train_samples: int):
        self.x, self.y = load_data3d(name)
        self.y_reduced = np.ma.masked_array(self.y, mask=(self.y == 0))
        self.labels = np.unique(self.y)[1:]
        # X_train, y_train, X_test, y_test = chose_samples(x, y, n_of_samples)
        self.x_train = list()
        self.y_train = np.empty((0,))
        self.x_test = list()
        self.y_test = np.empty((0,))
        self.train_indices = list()
        self.test_indices = list()
        self.select_samples(train_samples)
        self.y_test_predicted = np.empty((0,))
        self.y_predicted = np.zeros_like(self.y)
        print()

    def plot_ground_truth(self):
        plt.imshow(self.y_reduced)
        plt.colorbar()
        plt.show()

    def plot_test_set(self):
        y_train_mask = np.ma.getmask(self.y_reduced)
        for y, x in self.train_indices:
            y_train_mask[y, x] = 1
        y_train = np.ma.masked_array(self.y, mask=y_train_mask)
        plt.imshow(y_train)
        plt.colorbar()
        plt.show()

    def plot_train_set(self):
        y_train_mask = np.ones_like(self.y)
        for y, x in self.train_indices:
            y_train_mask[y, x] = 0
        y_train = np.ma.masked_array(self.y, mask=y_train_mask)

        plt.imshow(y_train)
        plt.colorbar()
        plt.show()

    def select_samples(self, amount: int):
        if amount == 0:
            samples_count_per_class = [self.y[self.y == label].size for label in self.labels]
            min_samples = min(samples_count_per_class)
            amount = int(min_samples/2)
        for label in self.labels:
            y_train, x_train = np.where(self.y_reduced == label)

            # test_mask = np.ones_like(self.y, dtype=bool)
            # for y, x in zip(y_train, x_train):
            #     test_mask[y, x] = 0
            # self.y_test = self.y[test_mask]
            # self.x_test = self.x[test_mask]

            selected_points = np.random.choice(x_train.size, amount, replace=False)
            self.y_train = np.append(self.y_train, [label] * amount)
            self.y_test = np.append(self.y_test, [label] * (y_train.size - amount))
            # for index in selected_points:
            #     selected_y, selected_x = y_train[index], x_train[index]
            #     train_indices.append((selected_y, selected_x))
            #     self.x_train.append(self.x[selected_y, selected_x, :])

            for index in range(x_train.size):
                selected_y, selected_x = y_train[index], x_train[index]
                if index in selected_points:
                    self.train_indices.append((selected_y, selected_x))
                    self.x_train.append(self.x[selected_y, selected_x, :])
                else:
                    self.test_indices.append((selected_y, selected_x))
                    self.x_test.append(self.x[selected_y, selected_x, :])

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)

    def scale_data(self):
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

    def reshape_data(self):
        self.x_train = self.x_train.reshape((self.x_train.shape[0], 1, self.x_train.shape[1], 1))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], 1, self.x_test.shape[1], 1))
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def predict(self, model: Model):
        self.y_test_predicted = model.predict(self.x_test)
        y_test_predicted = np.argmax(self.y_test_predicted, axis=1).astype(np.int8)
        i = 0
        mask = np.ones_like(self.y_predicted)
        for y, x in self.test_indices:
            self.y_predicted[y, x] = y_test_predicted[i]
            mask[y, x] = 0
            i += 1
        self.y_predicted = np.ma.masked_array(self.y_predicted, mask=mask)

    def plot_predictions(self):
        y_train_mask = np.ma.getmask(self.y_reduced)
        # y_predicted = np.ma.masked_array(self.y_predicted, mask=y_train_mask)
        plt.imshow(self.y_predicted)
        plt.colorbar()
        plt.show()

    def plot_all(self):
        plt.figure()
        plt.subplot(121)
        plt.gca().set_title('Ground Truth')
        plt.imshow(self.y_reduced, vmin=1, vmax=len(self.labels))
        plt.colorbar()

        plt.subplot(122)
        plt.gca().set_title('Predictions')
        # y_train_mask = np.ma.getmask(self.y_reduced)
        # y_predicted = np.ma.masked_array(self.y_predicted, mask=y_train_mask)
        plt.imshow(self.y_predicted, vmin=1, vmax=len(self.labels))
        plt.colorbar()
        plt.show()

    def postprocess(self, model, threshold: float=0.5):
        self.y_test_predicted = model.predict(self.x_test)
        good_predictions = self.y_test_predicted[np.max(self.y_test_predicted, axis=1) > 0.8].size/10
        # self.y_test_predicted = np.argmax(model.predict(self.x_test), axis=1).astype(np.int8)
        i = 0
        mask = np.ones_like(self.y_predicted)
        for y, x in self.test_indices:
            if self.y_test_predicted[i].max() > threshold or i == 0 or i == self.y_test_predicted.shape[0]-1:
                predicted_class = np.argmax(self.y_test_predicted[i]).astype(np.int8)
                self.y_predicted[y, x] = predicted_class
            else:
                starting_class = np.argmax(self.y_test_predicted[i]).astype(np.int8)
                # self.y_test_predicted[i] *= self.y_test_predicted[i-1] * self.y_test_predicted[i+1]
                self.y_test_predicted[i] = self.find_neighbors(i, y, x)

                # self.y_test_predicted[i] /= self.y_test_predicted[i].sum()
                # self.y_test_predicted[i] = np.argmax(self.y_test_predicted[i])
                predicted_class = np.argmax(self.y_test_predicted[i]).astype(np.int8)
                self.y_predicted[y, x] = predicted_class

            mask[y, x] = 0
            i += 1
        # self.y_predicted = np.ma.masked_array(self.y_predicted, mask=mask)
        good_predictions2 = self.y_test_predicted[np.max(self.y_test_predicted, axis=1) > 0.5].size / 10
        print()

    def find_neighbors(self, i: int, y: int, x: int):
        neighbors = self.y_test_predicted[i]
        for n_x in [x-1, x, x+1]:
            for n_y in [y - 1, y, y + 1]:
                if (n_y, n_x) in self.test_indices and not (n_x == x and n_y == y):
                    index = self.test_indices.index((n_y, n_x))
                    neighbors *= self.y_test_predicted[index]
                    neighbors /= neighbors.sum()
        return neighbors