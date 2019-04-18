import numpy as np
from scipy.io import loadmat


def create_sample_label_pairs(samples_by_class: list):
    all_sample_label_pairs = []
    for idx, class_ in enumerate(samples_by_class):
        for sample in class_:
            all_sample_label_pairs.append((sample, idx))
    np.random.shuffle(all_sample_label_pairs)
    samples, labels = zip(*all_sample_label_pairs)
    return np.array(samples), np.array(labels)


def produce_splits(samples: list, labels: np.ndarray, validation_size: float, test_size: float):
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
    Load data method.

    :param data_path: Path to data.
    :param ref_map_path: Path to labels.
    :return: Prepared data as a tuple.
    """
    data = None
    ref_map = None
    if data_path.endswith(".npy"):
        data = np.load(data_path)
    elif data_path.endswith(".mat"):
        mat = loadmat(data_path)
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
    non_zeros = np.nonzero(ref_map)
    prepared_data = []
    for i in range(data.shape[-1]):
        band = data[..., i][non_zeros]
        prepared_data.append(band)
    ref_map = ref_map[non_zeros]
    ref_map -= 1
    prepared_data = np.asarray(prepared_data).T
    return prepared_data, ref_map
