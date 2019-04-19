from copy import copy
from itertools import product
from math import ceil

from python_research.experiments.band_selection_algorithms.utils import *


def edge_preserving_filter(ref_map: np.ndarray, guided_image: np.ndarray,
                           neighborhood_size: int, epsilon: float = 1e-5):
    """
    Perform edge - preserving filtering on the newly created reference map.

    :param ref_map: Classification reference map.
    :param guided_image: Guided image as a mean over all bands from hyperspectral data.
    :param neighborhood_size: Size of the convolving window.
    :param epsilon: Regularizer constant.
    :return:
    """
    padding_size = neighborhood_size % ceil(float(neighborhood_size) / float(2))
    padded_cube = pad_zeros_3d(padding_size, ref_map)
    padded_guided_map = pad_zeros_2d(guided_image, padding_size)
    col_indexes, row_indexes = \
        list(range(ref_map.shape[ROW_AXIS])), list(range(ref_map.shape[COLUMNS_AXIS]))
    a_k_map, b_k_map = np.empty(shape=ref_map.shape), np.empty(shape=ref_map.shape)
    print("Calculating coefficients")
    for i in range(ref_map.shape[SPECTRAL_AXIS]):
        print("{}  band of one hot reference map".format(i))
        for row, col in product(col_indexes, row_indexes):
            p_k = copy(padded_cube[row:row + neighborhood_size,
                       col:col + neighborhood_size, i])
            i_k = copy(padded_guided_map[row:row + neighborhood_size,
                       col:col + neighborhood_size])
            sum_ = np.divide(np.sum(np.subtract(np.multiply(i_k, p_k), (np.mean(i_k) * np.mean(p_k)))),
                             neighborhood_size ** 2)
            a_k = sum_ / (np.var(i_k) + epsilon)
            b_k = np.mean(p_k) - a_k * np.mean(i_k)
            a_k_map[row, col, i] = a_k
            b_k_map[row, col, i] = b_k

    a_k_map = pad_zeros_3d(padding_size, a_k_map)
    b_k_map = pad_zeros_3d(padding_size, b_k_map)
    yi = np.empty(shape=ref_map.shape)
    row_indexes, col_indexes = \
        list(range(padding_size * 2, a_k_map.shape[ROW_AXIS] - (padding_size * 2))), \
        list(range(padding_size * 2, a_k_map.shape[COLUMNS_AXIS] - (padding_size * 2)))
    print("Obtaining new reference map")
    for i in range(ref_map.shape[SPECTRAL_AXIS]):
        print("{} band of new reference map".format(i))
        for x, y in product(row_indexes, col_indexes):
            x_indexes, y_indexes = \
                list(range(x - (padding_size * 2), x + 1)), \
                list(range(y - (padding_size * 2), y + 1)),
            a_k_sum = 0
            b_k_sum = 0
            for row, col in product(x_indexes, y_indexes):
                a_k_window = a_k_map[row:row + neighborhood_size, col:col + neighborhood_size, i]
                b_k_window = b_k_map[row:row + neighborhood_size, col:col + neighborhood_size, i]
                a_k_sum += np.sum(a_k_window)
                b_k_sum += np.sum(b_k_window)
            a_k_sum /= neighborhood_size ** 2
            b_k_sum /= neighborhood_size ** 2
            yi[x - padding_size * 2, y - padding_size * 2, i] = a_k_sum * guided_image[
                x - padding_size * 2, y - padding_size * 2] + b_k_sum

    new_ref_map = np.empty(shape=yi.shape[:SPECTRAL_AXIS])
    new_ref_map.fill(BG_CLASS)
    row_indexes, col_indexes = \
        list(range(new_ref_map.shape[ROW_AXIS])), list(range(new_ref_map.shape[COLUMNS_AXIS]))
    for row, col in product(row_indexes, col_indexes):
        new_ref_map[row, col] = np.argmax(yi[row, col]).astype(int)
    return new_ref_map


def pad_zeros_2d(guided_image, padding_size):
    """
    Pad zeros around 2D guided image.

    :param guided_image: Guided image.
    :param padding_size: Size of padding.
    :return: Padded image.
    """
    x = copy(guided_image)
    v_padding = np.zeros((padding_size, x.shape[1]))
    x = np.vstack((v_padding, x))
    x = np.vstack((x, v_padding))
    h_padding = np.zeros((x.shape[0], padding_size))
    x = np.hstack((h_padding, x))
    padded_guided_map = np.hstack((x, h_padding))
    return padded_guided_map


def pad_zeros_3d(padding_size, ref_map):
    """
    Pad zeros around 3D reference map.

    :param padding_size: size of padding.
    :param ref_map: Reference map.
    :return: Padded reference map.
    """
    x = copy(ref_map)
    v_padding = np.zeros((padding_size * 2, x.shape[1], x.shape[-1]))
    x = np.vstack((v_padding, x))
    x = np.vstack((x, v_padding))
    h_padding = np.zeros((x.shape[0], padding_size * 2, x.shape[-1]))
    x = np.hstack((h_padding, x))
    padded_cube = np.hstack((x, h_padding))
    return padded_cube
