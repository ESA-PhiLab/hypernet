import numpy as np
from scipy.io import loadmat

from python_research.experiments.band_selection_algorithms.utils import min_max_normalize_data


def create_sample_label_pairs(samples_by_class: list) -> tuple:
    """
    Return samples with labels.

    :param samples_by_class: List of samples divided by classes.
    :return: Prepared samples with labels.
    """
    all_sample_label_pairs = []
    for idx, class_ in enumerate(samples_by_class):
        for sample in class_:
            labels = np.zeros((len(samples_by_class)))
            labels[idx] = 1
            all_sample_label_pairs.append((sample, labels))
    np.random.shuffle(all_sample_label_pairs)
    samples, labels = zip(*all_sample_label_pairs)
    return np.array(samples), np.array(labels)


def produce_splits(samples: list, labels: np.ndarray, validation_size: float, test_size: float) -> tuple:
    """
    Divide data on sets based on passed validation and test sizes.

    :param samples: List of samples.
    :param labels: Array of labels.
    :param validation_size: Size of validation batch.
    :param test_size: Size of test batch.
    :return: Tuple of prepared samples with labels.
    """
    samples_per_class = [[] for _ in range(labels.max() + 1)]
    for x, y in zip(samples, labels):
        samples_per_class[y].append(x)
    lowest_class_population = len(samples_per_class[0])
    for class_ in samples_per_class:
        if len(class_) < lowest_class_population:
            lowest_class_population = len(class_)
    test_set_size = int(lowest_class_population * test_size)
    test_set = [[] for _ in range(len(samples_per_class))]
    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), test_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            test_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)
    validation_set_size = int(lowest_class_population * validation_size)
    validation_set = [[] for _ in range(len(samples_per_class))]
    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), validation_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            validation_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)
    training_set_size = int(lowest_class_population * (1 - (test_size + validation_size)))
    training_set = [[] for _ in range(len(samples_per_class))]
    for idx, class_ in enumerate(samples_per_class):
        chosen_indexes = np.random.choice(len(class_), training_set_size, replace=False)
        assert len(np.unique(chosen_indexes)) == len(chosen_indexes)
        for index in chosen_indexes:
            training_set[idx].append(class_[index])
        samples_per_class[idx] = np.delete(np.array(class_), [chosen_indexes], axis=0)
    return create_sample_label_pairs(training_set), \
           create_sample_label_pairs(validation_set), \
           create_sample_label_pairs(test_set)


def get_loader_function(data_path: str, ref_map_path: str) -> tuple:
    """
    Load data and perform min-max feature scaling.

    :param data_path: Path to data.
    :param ref_map_path: Path to labels.
    :return: Prepared data as a tuple.
    """
    data = None
    ref_map = None
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    if data_path.endswith(".mat"):
        mat = loadmat(data_path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    if ref_map_path.endswith(".npy"):
        ref_map = np.load(ref_map_path)
    if ref_map_path.endswith(".mat"):
        mat = loadmat(ref_map_path)
        for key in mat.keys():
            if "__" not in key:
                ref_map = mat[key]
                break
    assert data is not None and ref_map is not None, "The specified path or format of file is incorrect."
    data = min_max_normalize_data(data=data.astype(float))
    non_zeros = np.nonzero(ref_map)
    prepared_data = []
    for i in range(data.shape[-1]):
        band = data[..., i][non_zeros]
        prepared_data.append(band)
    ref_map = ref_map[non_zeros] - 1
    prepared_data = np.asarray(prepared_data).T
    return prepared_data, ref_map
