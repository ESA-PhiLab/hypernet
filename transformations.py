"""
File containg all the logic for loading and preparing the data to fit the model.
"""
from typing import Tuple

import numpy as np


def reshape_to_1D_samples(data: np.ndarray, 
                          labels: np.ndarray,
                          channels_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to [PIXEL, CHANNELS],
    so it fits the 1D Conv models
    :param data: Data to reshape.
    :param labels: Corresponding labels. 
    :param channels_idx: Index at which the channels are located in the provided data file
    :return: Reshape data and labels 
    :rtype: tuple with reshaped data and labels
    """
    data = data.reshape(data.shape[channels_idx], -1)
    data = np.moveaxis(data, -1, 0)
    labels = labels.reshape(-1)
    return data, labels