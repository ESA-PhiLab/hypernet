import functools
from itertools import product
from typing import Tuple, Union, List

import cv2
import numpy as np

import ml_intuition.data.io as io
import ml_intuition.enums as enums
from ml_intuition.data.utils import shuffle_arrays_together, \
    get_label_indices_per_class

HEIGHT = 0
WIDTH = 1
DEPTH = 2


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize labels so that they always start from 0
    :param labels: labels to normalize
    :return: Normalized labels
    """
    for label_index, label in enumerate(np.unique(labels)):
        labels[labels == label] = label_index
    return labels


def reshape_cube_to_2d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               channels_idx: int = 0,
                               use_unmixing: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to [PIXEL,CHANNELS, 1],
    so it fits the 2D Convolutional modules.
    :param data: Data to reshape.
    :param labels: Corresponding labels.
    :param channels_idx: Index at which the channels are located in the
                         provided data file.
    :param use_unmixing: Boolean indicating whether to perform experiments on the unmixing datasets,
            where classes in each pixel are present as fractions.
    :return: Reshape data and labels
    :rtype: tuple with reshaped data and labels
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, channels = data.shape
    data = data.reshape(height * width, channels)
    data = np.expand_dims(data, -1)
    if use_unmixing:
        labels = labels.reshape(height * width, -1)
    else:
        labels = labels.reshape(-1)

    data = data.astype(np.float32)
    if use_unmixing:
        labels = labels.astype(np.float32)
    else:
        labels = labels.astype(np.uint8)
    return data, labels


def get_padded_cube(data: np.ndarray, padding_size: int):
    v_padding = np.zeros((padding_size, data.shape[WIDTH], data.shape[DEPTH]))
    data = np.vstack((v_padding, data))
    data = np.vstack((data, v_padding))
    h_padding = np.zeros((data.shape[HEIGHT], padding_size, data.shape[DEPTH]))
    data = np.hstack((h_padding, data))
    data = np.hstack((data, h_padding))
    return data


def reshape_cube_to_3d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               neighborhood_size: int = 5,
                               background_label: int = 0,
                               channels_idx: int = 0,
                               use_unmixing: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data to a array of dimensionality: [N_SAMPLES, HEIGHT, WIDTH, CHANNELS]
    and the labels to dimensionality of: [N_SAMPLES, N_CLASSES]
    :param data: Data passed as array.
    :param labels: Labels passed as array.
    :param neighborhood_size: Length of the spatial patch.
    :param background_label: Label to filter out the background.
    :param channels_idx: Index of the channels.
    :param use_unmixing: Boolean indicating whether to perform experiments on the unmixing datasets,
            where classes in each pixel are present as fractions.
    :rtype: Tuple of data and labels reshaped to 3D format.
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, _ = data.shape
    padding_size = int(
        neighborhood_size % np.ceil(float(neighborhood_size) / 2.))
    data = get_padded_cube(data, padding_size)
    samples = []
    labels_3d = []
    data = data.astype(np.float32)
    for x, y in product(list(range(height)), list(range(width))):
        if use_unmixing or labels[x, y] != background_label:
            samples.append(data[x:x + padding_size * 2 + 1,
                           y:y + padding_size * 2 + 1])
            labels_3d.append(labels[x, y])
    samples = np.array(samples).astype(np.float32)
    if use_unmixing:
        labels3d = np.array(labels_3d).astype(np.float32)
    else:
        labels3d = np.array(labels_3d).astype(np.uint8)
    return samples, labels3d


def align_ground_truth(cube_2d_shape: Tuple[int, int], ground_truth: np.ndarray,
                       cube_to_gt_transform: np.ndarray) -> np.ndarray:
    """
    Align original labels to match the satellite hyperspectral cube using
    transformation matrix
    :param cube_2d_shape: Shape of the hyperspectral data cube
    :param ground_truth: Original labels as 2D array
    :param cube_to_gt_transform: Cube to ground truth transformation matrix
    :return: Transformed ground truth
    """
    gt_to_chan_transform = np.linalg.inv(cube_to_gt_transform)
    gt_transformed = cv2.warpPerspective(ground_truth, gt_to_chan_transform,
                                         cube_2d_shape, flags=cv2.INTER_NEAREST)
    return gt_transformed


def remove_nan_samples(data: np.ndarray, labels: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Remove samples which contain only nan values
    :param data: Data with dimensions [SAMPLES, ...]
    :param labels: Corresponding labels
    :return: Data and labels with removed samples containing nans
    """
    all_but_samples_axes = tuple(range(1, data.ndim))
    nan_samples_indexes = np.isnan(data).any(axis=all_but_samples_axes).ravel()
    labels = labels[~nan_samples_indexes]
    data = data[~nan_samples_indexes]
    return data, labels


def train_val_test_split(data: np.ndarray, labels: np.ndarray,
                         train_size: Union[List, float, int] = 0.8,
                         val_size: float = 0.1,
                         stratified: bool = True,
                         seed: int = 0,
                         use_unmixing: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage of each
                        class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param seed: Seed used for data shuffling
    :param use_unmixing: Boolean indicating whether to perform experiments on the unmixing datasets,
            where classes in each pixel are present as fractions.
    :return: train_x, train_y, val_x, val_y, test_x, test_y
    :raises AssertionError: When wrong type is passed as train_size
    """
    shuffle_arrays_together([data, labels], seed=seed)
    if use_unmixing:
        train_indices = np.random.choice(len(data), size=int(train_size * len(data)), replace=False)
        val_indices = np.random.choice(train_indices, size=int(val_size * len(train_indices)), replace=False)
    else:
        train_indices = _get_set_indices(train_size, labels, stratified)
        val_indices = _get_set_indices(val_size, labels[train_indices])
        val_indices = train_indices[val_indices]
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    train_indices = np.setdiff1d(train_indices, val_indices)
    return data[train_indices], labels[train_indices], data[val_indices], \
           labels[val_indices], data[test_indices], labels[test_indices]


@functools.singledispatch
def _get_set_indices(size: Union[List, float, int],
                     labels: np.ndarray,
                     stratified: bool = True) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    stratified parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0, if stratified = True, it
                    represents percentage of each class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    raise NotImplementedError()


@_get_set_indices.register(float)
def _(size: float,
      labels: np.ndarray,
      stratified: bool = True) -> np.ndarray:
    label_indices, unique_labels = get_label_indices_per_class(labels,
                                                               return_uniques=True)
    assert 0 < size <= 1
    if stratified:
        for idx in range(len(unique_labels)):
            samples_per_label = int(len(label_indices[idx]) * size)
            label_indices[idx] = label_indices[idx][:samples_per_label]
        train_indices = np.concatenate(label_indices, axis=0)
    else:
        train_indices = np.arange(int(len(labels) * size))
    return train_indices


@_get_set_indices.register(int)
def _(size: int,
      labels: np.ndarray,
      stratified: bool = True) -> np.ndarray:
    label_indices, unique_labels = get_label_indices_per_class(labels,
                                                               return_uniques=True)
    assert size >= 1
    if stratified:
        for label in range(len(unique_labels)):
            label_indices[label] = label_indices[label][:int(size)]
        train_indices = np.concatenate(label_indices, axis=0)
    else:
        train_indices = np.arange(size, dtype=int)
    return train_indices


@_get_set_indices.register(list)
def _(size: List,
      labels: np.ndarray,
      _) -> np.ndarray:
    label_indices, unique_labels = get_label_indices_per_class(labels,
                                                               return_uniques=True)
    if len(size) == 1:
        size = int(size[0])
    for n_samples, label in zip(size, range(len(unique_labels))):
        label_indices[label] = label_indices[label][:int(n_samples)]
    train_indices = np.concatenate(label_indices, axis=0)
    return train_indices


def patches_to_samples(data_file_path: str, neighborhood_size, val_size: float = 0.1, background_label: int = 0,
                       channels_idx=2):
    dataset = io.load_processed_h5(data_file_path)
    train_x = []
    train_y = []
    for patch, patch_gt in zip(dataset[enums.Dataset.TRAIN][enums.Dataset.DATA],
                               dataset[enums.Dataset.TRAIN][
                                   enums.Dataset.LABELS]):
        patch_samples, patch_samples_gt = reshape_cube_to_3d_samples(patch,
                                                                     patch_gt,
                                                                     neighborhood_size,
                                                                     background_label,
                                                                     channels_idx)
        train_x.append(patch_samples)
        train_y.append(patch_samples_gt)

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)

    test_x, test_y = reshape_cube_to_3d_samples(
        dataset[enums.Dataset.TEST][enums.Dataset.DATA],
        dataset[enums.Dataset.TEST][enums.Dataset.LABELS], neighborhood_size, background_label, channels_idx)

    train_y = train_y - 1
    test_y = test_y - 1
    val_indices = _get_set_indices(val_size, train_y)
    val_x = train_x[val_indices]
    val_y = train_y[val_indices]
    train_x = np.delete(train_x, val_indices, axis=0)
    train_y = np.delete(train_y, val_indices)
    return train_x, train_y, val_x, val_y, test_x, test_y
