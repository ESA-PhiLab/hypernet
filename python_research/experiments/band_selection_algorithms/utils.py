import numpy as np
from scipy.io import loadmat

BG_CLASS = -1
ROW_AXIS = 0
COLUMNS_AXIS = 1
SPECTRAL_AXIS = -1
CLASS_LABEL = 1


def load_data(data_path: str, ref_map_path: str) -> tuple:
    """
    Load data method.

    :param data_path: Path to data.
    :param ref_map_path: Path to labels.
    :return: Prepared data.
    """
    data = None
    ref_map = None
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    if data_path.endswith(".mat"):
        mat = loadmat(data_path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    if ref_map_path.endswith(".npy"):
        ref_map = np.load(ref_map_path)
    if ref_map_path.endswith(".mat"):
        mat = loadmat(ref_map_path)
        for key in mat.keys():
            if "__" not in key:
                ref_map = mat[key]
                break
    assert data is not None and ref_map is not None, "The specified path or format of file is incorrect."
    ref_map = ref_map.astype(int) + BG_CLASS
    return data.astype(float), ref_map.astype(int)


def min_max_normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Min-max data normalization method.

    :param data: Data cube.
    :return: Normalized data.
    """
    for band_id in range(data.shape[SPECTRAL_AXIS]):
        max_ = np.amax(data[..., band_id])
        min_ = np.amin(data[..., band_id])
        data[..., band_id] = (data[..., band_id] - min_) / (max_ - min_)
    return data


def mean_normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Mean normalization method.

    :param data: Data cube.
    :return: Normalized data.
    """
    for band_id in range(data.shape[SPECTRAL_AXIS]):
        max_ = np.amax(data[..., band_id])
        min_ = np.amin(data[..., band_id])
        mean = np.mean(data[..., band_id])
        data[..., band_id] = (data[..., band_id] - mean) / (max_ - min_)
    return data


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Data standardization method.

    :param data: Data cube.
    :return: Standardized data.
    """
    for band_id in range(data.shape[SPECTRAL_AXIS]):
        mean = np.mean(data[..., band_id])
        std = np.std(data[..., band_id])
        data[..., band_id] = (data[..., band_id] - mean) / std
    return data
