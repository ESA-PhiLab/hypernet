"""
All I/O related functions
"""

import csv
import os
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
from libtiff import TIFF
import tensorflow as tf

import ml_intuition.enums as enums


def save_metrics(dest_path: str, file_name: str, metrics: Dict[str, List]):
    """
    Save given dictionary of metrics.

    :param dest_path: Destination path.
    :param file_name: Name to save the file.
    :param metrics: Dictionary containing all metrics.
    """
    with open(os.path.join(dest_path, file_name), 'w') as file:
        write = csv.writer(file)
        write.writerow(metrics.keys())
        write.writerows(zip(*metrics.values()))


def extract_set(data_path: str, dataset_key: str) -> Dict[str, Union[np.ndarray, float]]:
    """
    Function for loading a h5 format dataset as a dictionary
        of samples, labels, min and max values.

    :param data_path: Path to the dataset.
    :param dataset_key: Key for dataset.
    """
    raw_data = h5py.File(data_path, 'r')
    dataset = {
        enums.Dataset.DATA: raw_data[dataset_key][enums.Dataset.DATA][:],
        enums.Dataset.LABELS: raw_data[dataset_key][enums.Dataset.LABELS][:],
        enums.DataStats.MIN: raw_data.attrs[enums.DataStats.MIN],
        enums.DataStats.MAX: raw_data.attrs[enums.DataStats.MAX]
    }
    raw_data.close()
    return dataset


def load_npy(data_file_path: str, gt_input_path: str) -> Tuple[
        np.ndarray, np.ndarray]:
    """
    Load .npy data and GT from specified paths

    :param data_file_path: Path to the data .npy file
    :param gt_input_path: Path to the GT .npy file
    :return: Tuple with loaded data and GT
    """
    return np.load(data_file_path), np.load(gt_input_path)


def load_satellite_h5(data_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hyperspectral cube and ground truth transformation matrix from .h5 file
    :param data_file_path: Path to the .h5 file
    :return: Hyperspectral cube and transformation matrix, both as np.ndarray
    """
    with h5py.File(data_file_path, 'r') as file:
        cube = file[enums.SatelliteH5Keys.CUBE][:]
        cube_to_gt_transform = file[enums.SatelliteH5Keys.GT_TRANSFORM_MAT][:]
    return cube, cube_to_gt_transform


def load_tiff(file_path: str) -> np.ndarray:
    """
    Load tiff image into np.ndarray
    :param file_path: Path to the .tiff file
    :return: Loaded image as np.ndarray
    """
    tiff = TIFF.open(file_path)
    return tiff.read_image()


def load_pb(path_to_pb: str) -> tf.GraphDef:
    """
    Load .pb file as a graph
    :param path_to_pb: Path to the .pb file
    :return: Loaded graph
    """
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
