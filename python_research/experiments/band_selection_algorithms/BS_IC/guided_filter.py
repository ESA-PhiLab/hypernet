from copy import copy
from itertools import product

from tqdm import tqdm

from python_research.experiments.band_selection_algorithms.utils import *


def edge_preserving_filter(ref_map: np.ndarray, guided_image: np.ndarray,
                           neighborhood_size: int, epsilon: float = 1e-10):
    """
    Perform edge - preserving filtering on the newly created reference map.

    :param ref_map: Classification reference map.
    :param guided_image: Guided image as a mean over all bands from hyperspectral data.
    :param neighborhood_size: Size of the convolving window.
    :param epsilon: Regularizer constant.
    :return:
    """
    col_indexes, row_indexes = \
        range(0, ref_map.shape[ROW_AXIS], neighborhood_size), range(0, ref_map.shape[COLUMNS_AXIS], neighborhood_size)
    print("Calculating coefficients:")
    a_k_map, b_k_map = np.empty(shape=ref_map.shape), np.empty(shape=ref_map.shape)
    for i in tqdm(range(ref_map.shape[SPECTRAL_AXIS]), total=ref_map.shape[SPECTRAL_AXIS]):
        for row, col in product(col_indexes, row_indexes):
            p_k = copy(ref_map[row:row + neighborhood_size, col:col + neighborhood_size, i])
            i_k = copy(guided_image[row:row + neighborhood_size, col:col + neighborhood_size])
            sum_ = np.sum(i_k * p_k - np.mean(i_k) * np.mean(p_k)) / (neighborhood_size ** 2)
            a_k = sum_ / (np.var(i_k) + epsilon)
            b_k = np.mean(p_k) - a_k * np.mean(i_k)
            a_k_map[row:row + neighborhood_size, col:col + neighborhood_size, i] = a_k
            b_k_map[row:row + neighborhood_size, col:col + neighborhood_size, i] = b_k

    yi = np.empty(shape=ref_map.shape)
    print("Calculating new \"improved\" reference map")
    for i in tqdm(range(ref_map.shape[SPECTRAL_AXIS]), total=ref_map.shape[SPECTRAL_AXIS]):
        for row_index, col_index in product(range(ref_map.shape[ROW_AXIS]),
                                            range(ref_map.shape[COLUMNS_AXIS])):
            a_k_sum, b_k_sum = 0, 0
            row_sub_indexes, col_sub_indexes = \
                list(range(row_index - neighborhood_size + 1, row_index + neighborhood_size)), \
                list(range(col_index - neighborhood_size + 1, col_index + neighborhood_size))
            for row, col in product(filter(lambda x: 0 <= x < ref_map.shape[ROW_AXIS], row_sub_indexes),
                                    filter(lambda x: 0 <= x < ref_map.shape[COLUMNS_AXIS], col_sub_indexes)):
                a_k_sum += a_k_map[row, col, i]
                b_k_sum += b_k_map[row, col, i]
            a_k_sum /= neighborhood_size ** 2
            b_k_sum /= neighborhood_size ** 2
            yi[row_index, col_index, i] = a_k_sum * guided_image[row_index, col_index] + b_k_sum
    new_ref_map = np.empty(shape=yi.shape[:-1])
    for row, col in product(range(new_ref_map.shape[ROW_AXIS]), range(new_ref_map.shape[COLUMNS_AXIS])):
        new_ref_map[row, col] = np.argmax(yi[row, col]).astype(int) + BG_CLASS
    return new_ref_map
