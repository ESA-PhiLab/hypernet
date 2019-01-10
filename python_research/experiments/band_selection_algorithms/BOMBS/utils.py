from python_research.experiments.band_selection_algorithms.BS_IC.utils import *


def prep_bands(selected_bands: np.ndarray) -> list:
    selected_bands = selected_bands.transpose()
    histograms = []
    for band in selected_bands:
        image_histogram = np.histogram(band, 256)[0] / band.size
        histograms.append(image_histogram)
    return histograms


def calculate_crowding_distances(list_of_antibodies: list):
    antibodies_entropy, antibodies_distance = get_fitness(
        list_of_antibodies=list_of_antibodies)
    crowding_distances = []
    for antibody_index in range(len(list_of_antibodies)):
        zeta_entropy = calculate_zeta_entropy(
            antigen_index=antibody_index,
            antibodies_entropy=antibodies_entropy,
            list_of_antibodies=list_of_antibodies) / (max(antibodies_entropy) - min(antibodies_entropy))
        zeta_distance = calculate_zeta_distance(
            antigen_index=antibody_index,
            antibodies_distance=antibodies_distance,
            list_of_antibodies=list_of_antibodies) / (max(antibodies_distance) - min(antibodies_distance))
        crowding_distances.append(zeta_entropy + zeta_distance)
    return np.asarray(crowding_distances)


def calculate_zeta_entropy(antigen_index, antibodies_entropy, list_of_antibodies):
    p_sorted_by_entropy = list(np.argsort(a=antibodies_entropy))
    if list_of_antibodies[antigen_index].L < list_of_antibodies[antigen_index].K:
        return 0
    if antigen_index == p_sorted_by_entropy[0] or antigen_index == p_sorted_by_entropy[-1]:
        return max(antibodies_entropy)
    else:
        left_antigen_index, right_antigen_index = \
            p_sorted_by_entropy[(p_sorted_by_entropy.index(antigen_index) - 1)], \
            p_sorted_by_entropy[(p_sorted_by_entropy.index(antigen_index) + 1)]
        return list_of_antibodies[right_antigen_index].entropy_fitness - \
               list_of_antibodies[left_antigen_index].entropy_fitness


def calculate_zeta_distance(antigen_index, antibodies_distance, list_of_antibodies):
    p_sorted_by_distance = list(np.argsort(a=antibodies_distance))
    if list_of_antibodies[antigen_index].L < list_of_antibodies[antigen_index].K:
        return 0
    if antigen_index == p_sorted_by_distance[0] or antigen_index == p_sorted_by_distance[-1]:
        return max(antibodies_distance)
    else:
        left_antigen_index, right_antigen_index = \
            p_sorted_by_distance[(p_sorted_by_distance.index(antigen_index) - 1)], \
            p_sorted_by_distance[(p_sorted_by_distance.index(antigen_index) + 1)]
        return list_of_antibodies[right_antigen_index].distance_fitness - \
               list_of_antibodies[left_antigen_index].distance_fitness


def get_fitness(list_of_antibodies: list):
    antibodies_entropy = [antibody.entropy_fitness for antibody in list_of_antibodies]
    antibodies_distance = [antibody.distance_fitness for antibody in list_of_antibodies]
    return [antibodies_entropy, antibodies_distance]


def load_data(path, ref_map_path, drop_bg=False):
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
    assert data is not None and ref_map_path is not None, 'There is no data to be loaded.'
    data = data.astype(float)
    min_ = np.amin(data)
    max_ = np.amax(data)
    data = (data - min_) / (max_ - min_)
    if drop_bg:
        non_zeros = np.nonzero(ref_map)
        prepared_data = []
        for i in range(data.shape[CONST_SPECTRAL_AXIS]):
            band = data[..., i][non_zeros]
            prepared_data.append(band)
        prepared_data = np.asarray(prepared_data).T
        return prepared_data
    else:
        prepared_data = []
        for i in range(data.shape[CONST_SPECTRAL_AXIS]):
            band = data[..., i]
            prepared_data.append(band.ravel())
        prepared_data = np.asarray(prepared_data).T
        return prepared_data
