"""
All I/O related functions
"""

import csv
import glob
import os
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import tensorflow as tf
import tifffile

import ml_intuition.enums as enums
from ml_intuition.data.utils import build_data_dict


def load_metrics(experiments_path: str, filename: str = None) -> \
        Dict[List, List]:
    """
    Load metrics to a dictionary.

    :param experiments_path: Path to the experiments directory.
    :param filename: Name of the file holding metrics. Defaults to
        'inference_metrics.csv'.
    :return: Dictionary containing all metric names and values from all experiments.
    """
    all_metrics = {'metric_keys': [], 'metric_values': []}
    for experiment_dir in glob.glob(
            os.path.join(experiments_path, '{}*'.format(enums.Experiment.EXPERIMENT))):
        if filename is None:
            inference_metrics_path = os.path.join(experiment_dir,
                               enums.Experiment.INFERENCE_METRICS)
        else:
            inference_metrics_path = os.path.join(experiment_dir, filename)
        with open(inference_metrics_path) as metric_file:
            reader = csv.reader(metric_file, delimiter=',')
            for row, key in zip(reader, all_metrics.keys()):
                all_metrics[key].append(row)
    return all_metrics


def save_metrics(dest_path: str, metrics: Dict[str, List], file_name: str=None):
    """
    Save given dictionary of metrics.

    :param dest_path: Destination path.
    :param file_name: Name to save the file.
    :param metrics: Dictionary containing all metrics.
    """
    if file_name is not None:
        dest_path = os.path.join(dest_path, file_name)
    with open(dest_path, 'w') as file:
        write = csv.writer(file)
        write.writerow(metrics.keys())
        write.writerows(zip(*metrics.values()))


def extract_set(data_path: str, dataset_key: str) -> Dict[str, Union[np.ndarray, float]]:
    """
    Function for loading a h5 format dataset as a dictionary
        of samples, labels, min and max values.

    :param data_path: Path to the dataset.
    :param dataset_key: Key for dataset.
    :return: Dictionary containing labels, data, min and max values.
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


def load_processed_h5(data_file_path: str) -> Dict:
    """
    Load procesed dataset containing the train, validation and test subsets
    with corresponding samples and labels.
    :param data_file_path: Path to the .h5 file.
    :return: Dictionary containing train, validation and test subsets.
    """
    with h5py.File(data_file_path, 'r') as file:
        train_x, train_y, val_x, val_y, test_x, test_y = \
            file[enums.Dataset.TRAIN][enums.Dataset.DATA][:], \
            file[enums.Dataset.TRAIN][enums.Dataset.LABELS][:],\
            file[enums.Dataset.VAL][enums.Dataset.DATA][:], \
            file[enums.Dataset.VAL][enums.Dataset.LABELS][:],\
            file[enums.Dataset.TEST][enums.Dataset.DATA][:], \
            file[enums.Dataset.TEST][enums.Dataset.LABELS][:]
    return build_data_dict(train_x=train_x, train_y=train_y,
                           val_x=val_x, val_y=val_y,
                           test_x=test_x, test_y=test_y)


def load_tiff(file_path: str) -> np.ndarray:
    """
    Load tiff image into np.ndarray
    :param file_path: Path to the .tiff file
    :return: Loaded image as np.ndarray
    """
    return tifffile.imread(file_path)


def save_md5(output_path, train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Save provided data as .md5 file
    :param output_path: Path to the filename
    :param train_x: Train set
    :param train_y: Train labels
    :param val_x: Validation set
    :param val_y: Validation labels
    :param test_x: Test set
    :param test_y: Test labels
    :return:
    """
    data_file = h5py.File(output_path, 'w')

    train_min, train_max = np.amin(train_x), np.amax(train_x)
    data_file.attrs.create(enums.DataStats.MIN, train_min)
    data_file.attrs.create(enums.DataStats.MAX, train_max)

    train_group = data_file.create_group(enums.Dataset.TRAIN)
    train_group.create_dataset(enums.Dataset.DATA, data=train_x)
    train_group.create_dataset(enums.Dataset.LABELS, data=train_y)

    val_group = data_file.create_group(enums.Dataset.VAL)
    val_group.create_dataset(enums.Dataset.DATA, data=val_x)
    val_group.create_dataset(enums.Dataset.LABELS, data=val_y)

    test_group = data_file.create_group(enums.Dataset.TEST)
    test_group.create_dataset(enums.Dataset.DATA, data=test_x)
    test_group.create_dataset(enums.Dataset.LABELS, data=test_y)
    data_file.close()


def read_min_max(path: str) -> Tuple[float, float]:
    """
    Read min and max value from a .csv file
    :param path:
    :return: Tuple with min and max
    """
    min_, max_ = np.loadtxt(path)
    return min_, max_


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


def save_confusion_matrix(matrix: np.ndarray, dest_path: str, filename: str = None):
    if filename is None:
        filename = 'confusion_matrix'
    np.savetxt(os.path.join(dest_path, filename + '.csv'),
               matrix, delimiter=',', fmt='%d')
