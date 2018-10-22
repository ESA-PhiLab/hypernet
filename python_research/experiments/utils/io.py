import os
from typing import Iterable
import numpy as np
from scipy.io import loadmat


def load_data(path: os.PathLike):
    """
    Loading data from NumPy array format (.npy) or from MATLAB format (.mat)
    :param path: Path to either .npy or .mat type file
    :return: numpy array with loaded data
    """
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    else:
        raise ValueError("This file type is not supported")
    return data


def save_to_csv(path: os.PathLike, to_save: Iterable, mode: str='a'):
    """
    Save an iterable to a CSV file
    :param path: Path to the file
    :param to_save: An iterable containing data to be saved
    :param mode: Mode in which the file will be opened
    :return: None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _, extension = os.path.splitext(path)
    if extension != ".csv" and extension != '':
        path.replace(extension, ".csv")
    elif extension == '':
        path += ".csv"
    csv = open(path, mode=mode)
    to_save_string = ",".join(str(x) for x in to_save) + "\n"
    csv.write(to_save_string)
    csv.close()
