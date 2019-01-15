import argparse

import numpy as np
from scipy.io import loadmat


def arg_parser():
    """
    Parse arguments for band selection algorithm.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for band selection based on improved classification map.')
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--ref_map_path', dest='ref_map_path', type=str)
    parser.add_argument('--dest_path', dest='dest_path', type=str)
    parser.add_argument('--neighborhood_size', dest='r', type=int, default=5)
    parser.add_argument('--training_patch', dest='training_patch', type=float, default=0.1)
    parser.add_argument('--bands_num', dest='bands_num', type=int)
    return parser.parse_args()


def load_data(path, ref_map_path, get_ref_map=True):
    """
    Load data method.

    :param path: Path to data.
    :param ref_map_path: Path to labels.
    :param get_ref_map: True if return reference map.
    :return: Prepared data.
    """
    data = None
    ref_map = None
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    if ref_map_path.endswith(".npy"):
        ref_map = np.load(ref_map_path)
    elif ref_map_path.endswith(".mat"):
        mat = loadmat(ref_map_path)
        for key in mat.keys():
            if "__" not in key:
                ref_map = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    assert data is not None and ref_map_path is not None, 'There is no data to be loaded.'
    min_ = np.amin(data)
    max_ = np.amax(data)
    data = (data - min_) / (max_ - min_)
    if get_ref_map is False:
        return data
    ref_map = ref_map.astype(int) + CONST_BG_CLASS
    return data.astype(float), ref_map.astype(int)


CONST_BG_CLASS = -1
CONST_ROW_AXIS = 0
CONST_COLUMNS_AXIS = 1
CONST_SPECTRAL_AXIS = -1
CONST_CLASS_LABEL = 1
INCREMENT_ONE = 1
SELECTED_BAND_FLAG = 0
