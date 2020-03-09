import os
import h5py
import numpy as np
import ast

import ml_intuition.data.utils as utils

dataset = os.environ.get('DATA_PATH')
input_node_name = ast.literal_eval(os.environ.get('INPUT_NODE_NAME'))
batch_size = 64


def calibrate_input(iter):
    with h5py.File(dataset, 'r') as file:
        train_data = file[utils.Dataset.TRAIN][utils.Dataset.DATA][()]
        train_data = train_data[:batch_size]
        train_data = np.expand_dims(train_data, axis=-1)
        train_min, train_max = file.attrs['min'], file.attrs['max']
    train_data = train_data - train_min / (train_max - train_min)
    return {input_node_name: train_data}
