from copy import copy
from itertools import product
from math import ceil, floor

from tqdm import tqdm

from python_research.experiments.band_selection_algorithms.utils import *


def edge_preserving_filter(ref_map: np.ndarray, guided_image: np.ndarray,
                           window_size: int, epsilon: float = 1e-10) -> np.ndarray:
    """
    Perform edge - preserving filtering on the newly created reference map.

    :param ref_map: Classification reference map.
    :param guided_image: Guided image as a mean over all bands from hyperspectral data.
    :param window_size: Size of the convolving window.
    :param epsilon: Regularizer constant.
    :return: Improved classification map.
    """
    print("Window size = {}".format(window_size))
    col_indexes, row_indexes = \
        range(0, ref_map.shape[ROW_AXIS], window_size), range(0, ref_map.shape[COLUMNS_AXIS], window_size)
    print("Calculating coefficients:")
    a_k_map, b_k_map = np.empty(shape=ref_map.shape), np.empty(shape=ref_map.shape)
    for i in tqdm(range(ref_map.shape[SPECTRAL_AXIS]), total=ref_map.shape[SPECTRAL_AXIS]):
        for row, col in product(col_indexes, row_indexes):
            p_k = copy(ref_map[row:row + window_size, col:col + window_size, i])
            i_k = copy(guided_image[row:row + window_size, col:col + window_size])
            sum_ = np.sum(i_k * p_k - np.mean(i_k) * np.mean(p_k)) / (window_size ** 2)
            a_k = sum_ / (np.var(i_k) + epsilon)
            b_k = np.mean(p_k) - a_k * np.mean(i_k)
            a_k_map[row:row + window_size, col:col + window_size, i] = a_k
            b_k_map[row:row + window_size, col:col + window_size, i] = b_k
    output_image = np.empty(shape=ref_map.shape)
    print("Calculating new \"improved\" classification map:")
    for i in tqdm(range(ref_map.shape[SPECTRAL_AXIS]), total=ref_map.shape[SPECTRAL_AXIS]):
        for row_index, col_index in product(range(ref_map.shape[ROW_AXIS]), range(ref_map.shape[COLUMNS_AXIS])):
            a_k_sum, b_k_sum = 0, 0
            row_sub_indexes, col_sub_indexes = \
                list(filter(lambda x: 0 <= x < ref_map.shape[ROW_AXIS],
                            list(range(row_index - floor(window_size / 2),
                                       row_index + ceil(window_size / 2))))), \
                list(filter(lambda x: 0 <= x < ref_map.shape[COLUMNS_AXIS],
                            list(range(col_index - floor(window_size / 2),
                                       col_index + ceil(window_size / 2)))))
            for sub_row_idx, sub_col_idx in product(row_sub_indexes, col_sub_indexes):
                a_k_sum += a_k_map[sub_row_idx, sub_col_idx, i]
                b_k_sum += b_k_map[sub_row_idx, sub_col_idx, i]
            a_k_sum, b_k_sum = a_k_sum / (row_sub_indexes.__len__() * col_sub_indexes.__len__()), \
                               b_k_sum / (row_sub_indexes.__len__() * col_sub_indexes.__len__())
            output_image[row_index, col_index, i] = a_k_sum * guided_image[row_index, col_index] + b_k_sum
    output_image = np.argmax(output_image, axis=-1) + BG_CLASS
    return output_image
