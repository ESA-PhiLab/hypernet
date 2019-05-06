import argparse
from typing import NamedTuple

from python_research.experiments.band_selection_algorithms.utils import *

ITER_RANGE = int(1e10)


class Arguments(NamedTuple):
    """
    Container for BOMBS band selection algorithm.
    """
    Gmax: int
    Na: int
    Nd: int
    Nc: int
    TD_size: int
    P_init_size: int
    bands_per_antibody: int
    data_path: str
    ref_map_path: str
    dest_path: str


def arguments() -> Arguments:
    """
    Arguments for running bombs selection algorithm.
    The default values are taken from the paper.
    """
    parser = argparse.ArgumentParser(description="Arguments for runner.")
    parser.add_argument("--Gmax", dest="Gmax", type=int, help="Maximum number of generations.", default=100)
    parser.add_argument("--Na", dest="Na", type=int, help="Maximum size of active population.", default=20)
    parser.add_argument("--Nd", dest="Nd", type=int, help="Maximum size of dominant population.", default=100)
    parser.add_argument("--Nc", dest="Nc", type=int, help="Maximum size of clone population.", default=100)
    parser.add_argument("--TD_size", dest="TD_size", type=int, help="Initial size of temporary dominant population.",
                        default=110)
    parser.add_argument("--P_init_size", dest="P_init_size", type=int,
                        help="Initial size of population P.", default=200)
    parser.add_argument("--bands_per_antibody", dest="bands_per_antibody",
                        type=int,
                        help="Number of selected bands.")
    parser.add_argument("--data_path", dest="data_path", type=str)
    parser.add_argument("--ref_map_path", dest="ref_map_path", type=str)
    parser.add_argument("--dest_path", dest="dest_path", type=str, help="Destination path for selected bands file.")
    return Arguments(**vars(parser.parse_args()))


def prep_bands(selected_bands: np.ndarray) -> list:
    """
    Prepare normalized gray-level histograms.

    :param selected_bands: Array containing selected bands.
    :return: List of all histograms.
    """
    selected_bands = selected_bands.transpose()
    histograms = []
    for band in selected_bands:
        image_histogram = np.histogram(band, 256)[0] / band.size
        histograms.append(image_histogram)
    return histograms


def calculate_crowding_distances(list_of_antibodies: list) -> np.ndarray:
    """
    Calculate crowding distances of each antibody.

    :param list_of_antibodies: List of individuals.
    :return: List of all crowding distances.
    """
    antibodies_entropy, antibodies_distance = get_fitness(
        list_of_antibodies=list_of_antibodies)
    crowding_distances = []
    arg_sorted_entropy, arg_sorted_distance = np.argsort(antibodies_entropy).tolist(), \
                                              np.argsort(antibodies_distance).tolist()
    for antibody_index in range(list_of_antibodies.__len__()):
        if list_of_antibodies[antibody_index].unique_band_size < list_of_antibodies[antibody_index].designed_band_size:
            crowding_distances.append(0)
            continue
        if antibody_index == arg_sorted_entropy[0] or antibody_index == arg_sorted_entropy[-1]:
            zeta_entropy = antibodies_entropy[arg_sorted_entropy[-1]] / (
                    antibodies_entropy[arg_sorted_entropy[-1]] -
                    antibodies_entropy[arg_sorted_entropy[0]])
        else:
            d_prime_prime = arg_sorted_entropy[arg_sorted_entropy.index(antibody_index) - 1]
            d_prime = arg_sorted_entropy[arg_sorted_entropy.index(antibody_index) + 1]
            min_diff = antibodies_entropy[d_prime] - antibodies_entropy[d_prime_prime]
            assert min_diff >= 0
            zeta_entropy = min_diff / (antibodies_entropy[arg_sorted_entropy[-1]] -
                                       antibodies_entropy[arg_sorted_entropy[0]])

        if antibody_index == arg_sorted_distance[0] or antibody_index == arg_sorted_distance[-1]:
            zeta_distance = antibodies_distance[arg_sorted_distance[-1]] / (
                    antibodies_distance[arg_sorted_distance[-1]] -
                    antibodies_distance[arg_sorted_distance[0]])
        else:
            d_prime_prime = arg_sorted_distance[arg_sorted_distance.index(antibody_index) - 1]
            d_prime = arg_sorted_distance[arg_sorted_distance.index(antibody_index) + 1]
            min_diff = antibodies_distance[d_prime] - antibodies_distance[d_prime_prime]
            assert min_diff >= 0
            zeta_distance = min_diff / (antibodies_distance[arg_sorted_distance[-1]] -
                                        antibodies_distance[arg_sorted_distance[0]])

        crowding_distances.append(zeta_entropy + zeta_distance)
    return np.asarray(crowding_distances)


def get_fitness(list_of_antibodies: list) -> list:
    """
    Return values of objective functions for each antibody.

    :param list_of_antibodies: List of individuals.
    :return: List of fitness scores.
    """
    antibodies_entropy = [antibody.entropy_fitness for antibody in list_of_antibodies]
    antibodies_distance = [antibody.distance_fitness for antibody in list_of_antibodies]
    return [antibodies_entropy, antibodies_distance]


def load_data(path: str, ref_map_path: str, drop_bg: bool = False) -> np.ndarray:
    """
    Load data method.

    :param path: Path to data.
    :param ref_map_path: Path to labels.
    :param drop_bg: True if background drop is intended.
    :return: Prepared data cube.
    """
    data = None
    ref_map = None
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    if ref_map_path.endswith(".npy"):
        ref_map = np.load(ref_map_path)
    elif ref_map_path.endswith(".mat"):
        mat = loadmat(ref_map_path)
        for key in mat.keys():
            if "__" not in key:
                ref_map = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    assert data is not None and ref_map_path is not None, "There is no data to be loaded."
    data = min_max_normalize_data(data=data.astype(float))
    if drop_bg:
        non_zeros = np.nonzero(ref_map)
        prepared_data = []
        for i in range(data.shape[SPECTRAL_AXIS]):
            band = data[..., i][non_zeros]
            prepared_data.append(band)
        prepared_data = np.asarray(prepared_data).T
        return prepared_data
    else:
        prepared_data = []
        for i in range(data.shape[SPECTRAL_AXIS]):
            band = data[..., i]
            prepared_data.append(band.ravel())
        prepared_data = np.asarray(prepared_data).T
        return prepared_data
