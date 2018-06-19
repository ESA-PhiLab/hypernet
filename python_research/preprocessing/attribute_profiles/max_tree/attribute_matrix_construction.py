import numpy as np
from .attributes_incrementally import StandardDeviation, LengthOfDiagonal, \
    FirstHuMoment, Area
from ..utils.data_types import Pixel


def construct_area_matrix(image: np.ndarray) -> np.ndarray:
    matrix = np.ones(image.shape, dtype=Area)
    image_width = image.shape[1]
    for index, _ in enumerate(image.flatten()):
        x = index % image_width
        y = int(index / image_width)
        matrix[y, x] = Area()
    return matrix


def construct_std_dev_matrix(image: np.ndarray) -> np.ndarray:
    image_width = image.shape[1]
    std_dev_matrix = np.zeros(image.shape, dtype=StandardDeviation)
    for index, pixel_value in enumerate(image.flatten()):
        x = index % image_width
        y = int(index / image_width)
        std_dev_matrix[y, x] = StandardDeviation(value=pixel_value)
    return std_dev_matrix


def construct_length_of_diagonal_matrix(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    image_size = image.size
    matrix = np.zeros(image.shape, dtype=LengthOfDiagonal)
    for index in range(0, image_size):
        x = index % width
        y = int(index / width)
        matrix[y, x] = LengthOfDiagonal(x, x, y, y)
    return matrix


def construct_first_hu_moment_matrix(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    max_ = float(np.amax(image))
    min_ = float(np.amin(image))
    matrix = np.zeros(image.shape, dtype=FirstHuMoment)
    for index, pixel_value in enumerate(image.flatten()):
        x = index % width
        y = int(index / width)
        norm_pixel_value = (float(pixel_value) - min_) / (max_ - min_)
        matrix[y, x] = FirstHuMoment(Pixel(x, y, norm_pixel_value))
    return matrix


matrix_constructs = {
    'area': construct_area_matrix,
    'stddev': construct_std_dev_matrix,
    'diagonal': construct_length_of_diagonal_matrix,
    'moment': construct_first_hu_moment_matrix
}


def construct_matrix(attribute_name: str, image: np.ndarray) -> np.ndarray:
    return matrix_constructs[attribute_name](image)
