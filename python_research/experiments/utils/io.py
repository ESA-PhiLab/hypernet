from os import PathLike

import numpy as np
from scipy.io import loadmat


def load_data(path: PathLike):
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    return data