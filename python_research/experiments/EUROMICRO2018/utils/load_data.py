import os
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

import matplotlib.pyplot as plt


try:
    PATH = ''
    os.listdir(PATH)
except:
    try:
        PATH = '/mnt/morpheus/Hyperspectral_Images'
        os.listdir(PATH)
    except:
        PATH = '/home/econib/jenkins/workspace/Hyperspectral_Imagining/Datasets'

DATA = {'indiana': 'Indian_pines_corrected.npy',
        'indian': 'Indian_pines_corrected.npy',
        'pavia': 'PaviaU_corrected.npy',
        'paviau': 'PaviaU_corrected.npy',
        'salinas': 'Salinas_corrected.npy'}


def load_data3d(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    x, y = load_data(dataset_name)
    return x, y

# def load_data2d(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
#     x, y = load_data(dataset_name)
#     # x, y = x.reshape((x.shape[0] * x.shape[1], x.shape[2])), y.reshape(y.shape[0] * y.shape[1])
#     # x, y = chose_samples(x, y, samples_per_class)
#
#     return x.reshape((x.shape[0] * x.shape[1], x.shape[2])), y.reshape(y.shape[0] * y.shape[1])


def load_data2d(dataset_name: str,
                samples_per_class: int=None) -> Tuple:
    x, y = load_data(dataset_name)
    x, y = x.reshape((x.shape[0] * x.shape[1], x.shape[2])), y.reshape(y.shape[0] * y.shape[1])

    return x, y


# def train_test_split(data: np.ndarray, labels: np.ndarray,
#                      samples_per_class: int=None, ratio: float=None):
#
#     x, y = chose_samples(data, labels, samples_per_class)


def chose_samples(data: np.ndarray,
                  labels: np.ndarray,
                  samples_per_class: int=0) -> Tuple:
    label_count = len(np.unique(labels))
    samples_count_per_class = [labels[labels == label].size for label in range(label_count)]
    if samples_per_class == 0:
        min_samples = int(min(samples_count_per_class)/2)
    else:
        min_samples = samples_per_class

    x_train = list()
    x_test = list()
    y_train = list([])
    y_test = list([])
    for label in range(1, label_count):
        mask = np.array(np.zeros_like(labels))
        mask[labels == label] = 1
        p = mask.astype(np.float)/mask.sum()
        idx_train = np.random.choice(range(data.shape[0]), min_samples, replace=False, p=p)

        test_mask = labels == label
        test_mask[idx_train] = False
        idx_test = np.arange(data.shape[0])[test_mask]

        x_train.append(data[idx_train])
        x_test.append(data[idx_test])
        y_train += [label] * min_samples
        y_test += [label] * (samples_count_per_class[label]-min_samples)

    x_train = np.asarray(x_train)
    x_train = x_train.reshape((x_train.shape[0] * x_train.shape[1], x_train.shape[2]))
    x_test = np.vstack(x_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    return tuple(shuffle(x_train, y_train)) + tuple(shuffle(x_test, y_test))


def load_data(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    if dataset_name.lower() in DATA.keys():
        data_path = os.path.join(PATH, DATA[dataset_name])
        ground_truth_path = os.path.join(PATH, DATA[dataset_name].replace('corrected', 'gt'))
        data = np.load(data_path)
        return data.astype(np.float), np.load(ground_truth_path)

    elif os.path.exists(dataset_name):
        data = np.load(dataset_name)
        ground_truth = np.load(dataset_name.replace('corrected', 'gt'))
        return data.transpose((2, 0, 1)).astype(np.float), ground_truth



