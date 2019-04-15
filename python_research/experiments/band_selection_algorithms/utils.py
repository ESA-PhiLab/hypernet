import numpy as np
from scipy.io import loadmat

BG_CLASS = -1
ROW_AXIS = 0
COLUMNS_AXIS = 1
SPECTRAL_AXIS = -1
CLASS_LABEL = 1


def load_data(data_path, ref_map_path, get_ref_map=True):
    """
    Load data method.

    :param data_path: Path to data.
    :param ref_map_path: Path to labels.
    :param get_ref_map: True if return reference map.
    :return: Prepared data.
    """
    data = None
    ref_map = None
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    elif data_path.endswith(".mat"):
        mat = loadmat(data_path)
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
    ref_map = ref_map.astype(int) + BG_CLASS
    return data.astype(float), ref_map.astype(int)
