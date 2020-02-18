"""
All I/O related functions
"""

from typing import Tuple
import numpy as np


def load_npy(data_input_path: str, gt_input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load .npy data and GT from specified paths
    
    :param data_input_path: Path to the data .npy file
    :param gt_input_path: Path to the GT .npy file
    :return: Tuple with loaded data and GT
    """
    return np.load(data_input_path), np.load(gt_input_path)