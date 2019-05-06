import argparse
import os
from itertools import product

from sklearn import svm

from python_research.experiments.band_selection_algorithms.icm.guided_filter import edge_preserving_filter
from python_research.experiments.band_selection_algorithms.utils import *


def prepare_datasets(ref_map: np.ndarray, training_patch: float) -> tuple:
    """
    Prepare data for band selection.

    :param ref_map: Reference map containing labels.
    :param training_patch: Patch containing training data.
    :return: Returns prepared data in tuple.
    """
    samples_by_classes = [[] for _ in range(int(ref_map.max()) + abs(BG_CLASS))]
    rows, columns = list(range(int(ref_map.shape[ROW_AXIS]))), list(range((int(ref_map.shape[COLUMNS_AXIS]))))
    for i, j in product(rows, columns):
        if ref_map[i, j] != BG_CLASS:
            samples_by_classes[ref_map[i, j]].append([[i, j], ref_map[i, j]])
    lowest_class_population = np.min([samples_by_classes[i].__len__() for i in range(samples_by_classes.__len__())])
    train_set, test_set = [], []
    train_size = int(training_patch * lowest_class_population)
    for i in range(samples_by_classes.__len__()):
        train_set.extend(samples_by_classes[i][:train_size])
        test_set.extend(samples_by_classes[i][train_size:])
    train_samples, train_labels = list(map(list, zip(*train_set)))
    test_samples, test_labels = list(map(list, zip(*test_set)))
    return train_samples, np.asarray(train_labels), test_samples, np.asarray(test_labels)


def get_data_by_indexes(indexes: list, data: np.ndarray) -> np.ndarray:
    """
    Return data block given indexes.

    :param indexes: Indexes of samples.
    :param data: Hyperspectral data block.
    :return: Loaded samples.
    """
    return np.asarray([data[i, j] for i, j in indexes])


def one_hot_map(ref_map: np.ndarray) -> np.ndarray:
    """
    Perform one-hot encoding over new reference map.

    :param ref_map: Passed reference map.
    :return: One-hot encoded reference map.
    """
    ref_map += abs(BG_CLASS)
    one_hot_ref_map = np.zeros(shape=[ref_map.shape[ROW_AXIS], ref_map.shape[COLUMNS_AXIS],
                                      ref_map.max() + abs(BG_CLASS)])
    rows, columns = list(range(ref_map.shape[ROW_AXIS])), list(range((ref_map.shape[COLUMNS_AXIS])))
    for i, j in product(rows, columns):
        one_hot_ref_map[i, j, ref_map[i, j].astype(int)] = CLASS_LABEL
    return one_hot_ref_map


def get_guided_image(data: np.ndarray) -> np.ndarray:
    """
    Return the guided image as a mean array over all bands.

    :param data: Hyperspectral data block.
    :return: Mean array over all bands.
    """
    return np.mean(data, axis=SPECTRAL_AXIS)


def construct_new_ref_map(labels: np.ndarray, samples: list, ref_map_shape: list):
    """
    Based on the SVM predictions, create new reference map.

    :param labels: Labels for samples for constructing new reference map.
    :param samples: Indexes of samples for constructing new reference map.
    :param ref_map_shape: Designed shape of the new reference map.
    :return: New reference map based on the classifier prediction.
    """
    new_ref_map = np.zeros(shape=ref_map_shape) + BG_CLASS
    for i, indexes in enumerate(samples):
        new_ref_map[indexes[ROW_AXIS], indexes[COLUMNS_AXIS]] = labels[i]
    return new_ref_map.astype(int)


def train_svm(data: np.ndarray, test_labels: list, test_samples: list, train_labels: list,
              train_samples: list) -> np.ndarray:
    """
    Train SVM on input data and return its predictions.
    During band selection process, parameters of SVM are fixed in order to reduce computation burden.

    :param data: Hyperspectral data block.
    :param test_labels: Test labels.
    :param test_samples: Indexes of test samples.
    :param train_labels: Train labels.
    :param train_samples: Indexes of train samples.
    :return: Prediction which is used to create new reference map.
    """
    model = svm.SVC(kernel="rbf", C=1024, gamma=2)
    model.fit(get_data_by_indexes(train_samples, data), train_labels)
    prediction = model.predict(get_data_by_indexes(test_samples, data))
    print("SVM fitness score {0:5.2f}%".format(
        model.score(get_data_by_indexes(test_samples, data), test_labels) * float(100)))
    return prediction


def generate_pseudo_ground_truth_map(args: argparse.Namespace):
    """
    Generate and save pseudoground truth map which is used in the band selection process.

    :param args: Parsed arguments.
    """
    data, ref_map = load_data(data_path=args.data_path, ref_map_path=args.ref_map_path)
    data = min_max_normalize_data(data=data)
    guided_image = get_guided_image(data=data)
    train_samples, train_labels, test_samples, test_labels = prepare_datasets(ref_map=ref_map,
                                                                              training_patch=args.training_patch)

    prediction = train_svm(data=data, test_labels=test_labels, test_samples=test_samples,
                           train_labels=train_labels, train_samples=train_samples)

    updated_ref_map = construct_new_ref_map(labels=np.concatenate((train_labels, prediction)),
                                            samples=train_samples + test_samples,
                                            ref_map_shape=ref_map.shape)

    one_hot_ref_map = one_hot_map(ref_map=updated_ref_map.copy())

    print("SVM classification map similarity score according to GT map {0:5.2f}%".format(
        ((ref_map == updated_ref_map).sum() / ref_map.size) * float(100)))

    window_size = 2 * (args.radius_size - 1) + 1

    improved_class_map = edge_preserving_filter(ref_map=one_hot_ref_map,
                                                window_size=window_size,
                                                guided_image=guided_image)

    np.save(os.path.join(args.dest_path, "improved_classification_map_{}".format(str(args.bands_num))),
            improved_class_map)
