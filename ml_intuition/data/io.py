"""
All I/O related functions
"""

import csv
import os
from typing import Dict, Tuple

import h5py
import numpy as np

from ml_intuition.data import utils


def save_metrics(dest_path: str, metric_key: str, metrics: Dict):
    """
    Save given dictionary of metrics.

    :param dest_path: Destination path.
    :param metric_key: Key to save the file.
    :param metrics: Dictionary containing all metrics.
    """
    with open(os.path.join(dest_path, metric_key), 'w') as file:
        write = csv.writer(file)
        write.writerow(metrics.keys())
        write.writerows(zip(*metrics.values()))


def load_data(data_path: str, dataset_key: str) -> Dict[np.ndarray, np.ndarray]:
    """
    Function for loading a h5 format dataset as a dictionary of samples and labels.

    :param data_path: Path to the dataset.
    :param keys: Key for dataset.
    """
    raw_data = h5py.File(data_path, 'r')
    dataset = {
        utils.Dataset.DATA: np.asarray(
            raw_data[dataset_key][utils.Dataset.DATA]),
        utils.Dataset.LABELS: np.asarray(
            raw_data[dataset_key][utils.Dataset.LABELS]),
    }
    min_ = raw_data.attrs['min']
    max_ = raw_data.attrs['max']
    raw_data.close()
    return dataset, min_, max_


def load_npy(data_input_path: str, gt_input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load .npy data and GT from specified paths

    :param data_input_path: Path to the data .npy file
    :param gt_input_path: Path to the GT .npy file
    :return: Tuple with loaded data and GT
    """
    return np.load(data_input_path), np.load(gt_input_path)
