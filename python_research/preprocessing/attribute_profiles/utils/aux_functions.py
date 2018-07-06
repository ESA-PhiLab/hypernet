import numpy as np
from copy import copy
from sklearn.decomposition import PCA
from ..utils.data_types import StdDevIncrementally, Pixel


def construct_std_dev_matrix(image):
    image_width = image.shape[1]
    std_dev_matrix = np.zeros(image.shape, dtype=StdDevIncrementally)
    for index, pixel_value in enumerate(image.flatten()):
        x = index % image_width
        y = int(index / image_width)
        std_dev_matrix[y, x] = StdDevIncrementally(value=pixel_value)
    return std_dev_matrix


def radix_sort(array, base=10):
    def list_to_buckets(array, base, iteration):
        buckets = [[] for _ in range(base)]
        for number in array:
            digit = (number.gray_level // (base ** iteration)) % base
            buckets[int(digit)].append(number)
        return buckets

    def buckets_to_list(buckets):
        numbers = []
        for bucket in buckets:
            for number in bucket:
                numbers.append(number)
        return numbers

    maxval = max(array).gray_level

    it = 0
    while base ** it <= maxval:
        array = buckets_to_list(list_to_buckets(array, base, it))
        it += 1

    return array


def push_root(root: Pixel, array):
    for index, s_pixel in enumerate(array):
        if root.gray_level < root.gray_level:
            array.insert(index, root)
            return array
    array.append(root)
    return array


def calculate_pca(x, n_components):
    original_shape = x.shape
    x = x.reshape((original_shape[0] * original_shape[1], original_shape[2]))
    pca = PCA(n_components=n_components).fit_transform(x)
    pca = pca.reshape((original_shape[0], original_shape[1], n_components))
    return pca


def normalize_pca(pc, lower_limit, upper_limit):
    pc_min = np.amin(pc)
    pc_max = np.amax(pc)
    pc = ((upper_limit - lower_limit) * (pc - pc_min)) / (pc_max - pc_min) + \
         lower_limit
    return pc


def invert_array(x):
    max_ = np.amax(x)
    x = max_ - x
    return x


def thickening_area(x, tree, areas):
    area_thinned = np.zeros((x.shape[0], x.shape[1], len(areas)))
    for index, area in enumerate(areas):
        img_a = tree.filter('area', area)
        area_thinned[:, :, index] = copy(img_a)
    return area_thinned


def thickening_std(x, tree, stds):
    std_thinned = np.zeros((x.shape[0], x.shape[1], len(stds)))
    for index, std in enumerate(stds):
        img = tree.filter('stdev', std)
        std_thinned[:, :, index] = copy(img)
    return std_thinned


def thinning_std(x, tree, stds):
    std_thinned = np.zeros((x.shape[0], x.shape[1], len(stds)))
    for index, std in enumerate(stds):
        img = tree.filter('stdev', std)
        img = invert_array(img)
        std_thinned[:, :, index] = copy(img)
        index += 1
    return std_thinned


def thinning_area(x, tree, areas):
    area_thinned = np.zeros((x.shape[0], x.shape[1], len(areas)))
    for index, area in enumerate(areas):
        img_a = tree.filter('area', area)
        img_a = invert_array(img_a)
        area_thinned[:, :, index] = copy(img_a)
    return area_thinned
