import numpy as np
from copy import copy
from typing import List, Tuple
from .utils.aux_functions import invert_array_values


def attribute_thickening(
    tree,
    attribute: str,
    thresholds: List,
    shape_2d: Tuple
) -> np.ndarray:
    thickened = np.zeros(shape_2d + (len(thresholds), ))
    for index, threshold in enumerate(thresholds):
        filtered = tree.filter(attribute, threshold)
        thickened[:, :, index] = copy(filtered)
    return thickened


def attribute_thinning(
    tree,
    attribute: str,
    thresholds: List,
    shape_2d: Tuple
) -> np.ndarray:
    thinned = np.zeros(shape_2d + (len(thresholds), ))
    for index, threshold in enumerate(thresholds):
        filtered = tree.filter(attribute, threshold)
        filtered = invert_array_values(filtered)
        thinned[:, :, index] = copy(filtered)
    return thinned


def generate_ap(
    image: np.ndarray,
    tree_thick,
    tree_thin,
    attribute: str,
    thresholds: List
) -> np.ndarray:
    shape_2d = image.shape
    eap = []
    thickened = attribute_thickening(tree_thick, attribute, thresholds, shape_2d)
    thinned = attribute_thinning(tree_thin, attribute, thresholds, shape_2d)
    for ap in range(0, thickened.shape[-1]):
        eap.append(thickened[:, :, ap])

    eap.append(image)

    for ap in range(0, thinned.shape[-1]):
        eap.append(thinned[:, :, ap])

    return np.array(eap).swapaxes(0, 2).swapaxes(0, 1)
