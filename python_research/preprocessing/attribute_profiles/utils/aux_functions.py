import numpy as np
from sklearn.decomposition import PCA


def calculate_pca(x, n_components):
    original_shape = x.shape
    x = x.reshape((original_shape[0] * original_shape[1], original_shape[2]))
    pca = PCA(n_components=n_components).fit_transform(x)
    return pca.reshape((original_shape[0], original_shape[1], n_components))


def normalize_pca(pc, lower_limit, upper_limit):
    pc_min = np.amin(pc)
    pc_max = np.amax(pc)
    return ((upper_limit - lower_limit) * (pc - pc_min)) / (pc_max - pc_min) + lower_limit


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

    maxval = max(array).value

    it = 0
    while base ** it <= maxval:
        array = buckets_to_list(list_to_buckets(array, base, it))
        it += 1

    return array


def invert_array(x):
    return np.amax(x) - x
