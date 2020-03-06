from itertools import product
from typing import Tuple

import numpy as np

from ml_intuition.data.utils import Coordinates

NORMALIZE_VALUE = 2
import cv2


def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize labels so that they always start from 0
    :param labels: labels to normalize
    :return: Normalized labels
    """
    min_label = np.amin(labels)
    return labels - min_label


def reshape_cube_to_2d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               channels_idx: int = 0) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to [PIXEL,
    CHANNELS, 1], so it fits the 2D Conv models
    :param data: Data to reshape.
    :param labels: Corresponding labels.
    :param channels_idx: Index at which the channels are located in the
                         provided data file
    :return: Reshape data and labels
    :rtype: tuple with reshaped data and labels
    """
    data = data.reshape(data.shape[channels_idx], -1)
    data = np.moveaxis(data, -1, 0)
    data = np.expand_dims(data, -1)
    labels = labels.reshape(-1)
    return data, labels


def align_ground_truth(cube_2d_shape: Tuple[int, int], labels: np.ndarray,
                       chan_to_gt_transform: np.ndarray) -> np.ndarray:
    """
    Align original labels to match the satellite hyperspectral cube using
    transformation matrix
    :param cube_2d_shape: Shape of the hyperspectral data cube
    :param labels: Original labels as 2D array
    :param chan_to_gt_transform: Ground truth transformation matrix used to
                          transform coordinates from hyperspectral cube to
                          corresponding label coordinates
    :return: Aligned labels
    """
    gt_to_chan_transform = np.linalg.inv(chan_to_gt_transform)
    gt_transformed = cv2.warpPerspective(labels, gt_to_chan_transform,
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
    data = data[~nan_samples_indexes, ...]
    return data, labels
