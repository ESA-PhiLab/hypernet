"""
All I/O related functions
"""

import h5py
from typing import Tuple
import numpy as np

from libtiff import TIFF

from ml_intuition.data.utils import SatelliteH5Keys


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
        cube = file[SatelliteH5Keys.CUBE][()]
        cube_to_gt_transform = file[SatelliteH5Keys.GT_TRANSFORM_MAT][()]
    return cube, cube_to_gt_transform


def load_tiff(file_path: str) -> np.ndarray:
    """
    Load tiff image into np.ndarray
    :param file_path: Path to the .tiff file
    :return: Loaded image as np.ndarray
    """
    tiff = TIFF.open(file_path)
    return tiff.read_image()
