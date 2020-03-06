"""
All I/O related functions
"""

from typing import Dict, Tuple

import h5py
import numpy as np

from ml_intuition.data import utils


def load_data(data_path: str, dataset_key: str) -> Dict[np.ndarray, np.ndarray]:
    """
    Function for loading a given dataset as a dictionary of samples and labels.

    :param data_path: Path to the dataset.
    :param keys: Key for dataset.
    """
    raw_data = h5py.File(data_path, 'r')
    dataset = {
        utils.Dataset.DATA: np.asarray(raw_data[dataset_key][utils.Dataset.DATA]),
        utils.Dataset.LABELS: np.asarray(
            raw_data[dataset_key][utils.Dataset.LABELS])
    }
    raw_data.close()
    return dataset


def load_npy(data_input_path: str, gt_input_path: str) -> Tuple[
        np.ndarray, np.ndarray]:
    """
    Load .npy data and GT from specified paths

    :param data_input_path: Path to the data .npy file
    :param gt_input_path: Path to the GT .npy file
    :return: Tuple with loaded data and GT
    """
    return np.load(data_input_path), np.load(gt_input_path)
