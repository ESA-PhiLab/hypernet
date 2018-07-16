import numpy as np
from copy import copy
from sklearn.decomposition import PCA
from typing import List, Tuple


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
