"""
File containing functions which calibrate input for frozen graphs used for
quantization. All arguments to such functions are passed through the shell
environment.
"""
import os
from typing import Dict
import h5py
import numpy as np

import ml_intuition.data.utils as utils

dataset = os.environ.get('DATA_PATH')
input_node_name = os.environ.get('INPUT_NODE_NAME')
batch_size = int(os.environ.get('BATCH_SIZE'))


def calibrate_2d_input(iter: int) -> Dict[str, np.ndarray]:
    """
    Return dictionary with a batch of the training data to be used for
    quantization calibration. One dimension and the end is added and the
    min max normalization is performed.
    :param iter: Int object indicating the calibration step number
    :return: Dict with name of the input node as key and training samples as
             np.ndarray
    """
    batch_start = iter * batch_size
    batch_end = iter * batch_size + batch_size
    with h5py.File(dataset, 'r') as file:
        train_data = file[utils.Dataset.TRAIN][utils.Dataset.DATA][()]
        train_data = train_data[batch_start:batch_end]
        train_data = np.expand_dims(train_data, axis=-1)
        train_min, train_max = file.attrs['min'], file.attrs['max']
    train_data = (train_data - train_min) / (train_max - train_min)
    return {input_node_name: train_data}
