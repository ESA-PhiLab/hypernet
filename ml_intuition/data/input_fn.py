"""
File containing functions which calibrate input for frozen graphs used for
quantization. All arguments to such functions are passed through the shell
environment, because they are executed through the internal decent tools.
"""
import os
from typing import Dict
import h5py
import numpy as np

import ml_intuition.enums as enums

dataset = os.environ.get('DATA_PATH')
input_node_name = os.environ.get('INPUT_NODE_NAME')
batch_size = int(os.environ.get('BATCH_SIZE'))


def calibrate_2d_input(iter: int) -> Dict[str, np.ndarray]:
    """
    Return dictionary with a batch of the training data to be used for
    quantization calibration. One dimension and the end is added and the
    min max normalization is performed.
    :param iter: Int object indicating the calibration step
    :return: Dict with name of the input node as key and training samples as
             np.ndarray
    """
    batch_start = iter * batch_size
    batch_end = iter * batch_size + batch_size
    with h5py.File(dataset, 'r') as file:
        samples = file[enums.Dataset.TRAIN][enums.Dataset.DATA][()]
        samples = samples[batch_start:batch_end]
        samples = np.expand_dims(samples, axis=-1)
        min_value, max_value = file.attrs[enums.DataStats.MIN], \
                               file.attrs[enums.DataStats.MAX]
    samples = (samples - min_value) / (max_value - min_value)
    return {input_node_name: samples}
