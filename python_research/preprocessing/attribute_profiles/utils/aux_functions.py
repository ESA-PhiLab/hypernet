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


def invert_array(x):
    return np.amax(x) - x
