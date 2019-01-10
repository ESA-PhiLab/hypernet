import os
from itertools import product
from random import shuffle

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from python_research.experiments.band_selection_algorithms.BS_IC.guided_filter import edge_preserving_filter
from python_research.experiments.band_selection_algorithms.BS_IC.utils import *


def prepare_datasets(ref_map: np.ndarray, training_patch: float, do_shuffle: bool = False) -> tuple:
    """
    Prepare data for SVMs training.

    :param ref_map: Reference map for labels.
    :param training_patch: Patch for training data, concern the lowest population class.
    :param do_shuffle: True if shuffle is considered.
    :return: Returns prepared data in tuple.
    """
    print('Training data patch: {}'.format(training_patch))
    samples_by_classes = [[] for _ in range(ref_map.max() + abs(CONST_BG_CLASS))]
    rows, columns = list(range(ref_map.shape[CONST_ROW_AXIS])), list(range((ref_map.shape[CONST_COLUMNS_AXIS])))
    for i, j in product(rows, columns):
        if ref_map[i, j] != CONST_BG_CLASS:
            samples_by_classes[ref_map[i, j]].append([[i, j], ref_map[i, j]])
    lowest_class_population = samples_by_classes[0].__len__()
    for i in range(1, samples_by_classes.__len__()):
        if lowest_class_population > samples_by_classes[i].__len__():
            lowest_class_population = samples_by_classes[i].__len__()
    train_set, test_set = [], []
    train_size = int(training_patch * lowest_class_population)
    for i in range(samples_by_classes.__len__()):
        train_set.extend(samples_by_classes[i][:train_size])
        test_set.extend(samples_by_classes[i][train_size:])
    if do_shuffle:
        shuffle(train_set)
        shuffle(test_set)
    train_samples, train_labels = list(map(list, zip(*train_set)))
    test_samples, test_labels = list(map(list, zip(*test_set)))
    return train_samples, np.asarray(train_labels), test_samples, np.asarray(test_labels)


def get_data_by_indexes(indexes, data) -> np.ndarray:
    """
    Return data block given indexes.

    :param indexes: List of indexes.
    :param data: Hyperspectral data block.
    :return: Chosen data blocks.
    """
    train_samples = []
    for i, j in indexes:
        train_samples.append(data[i, j])
    return np.asarray(train_samples)


def one_hot_map(ref_map):
    max_ = (ref_map.max() + abs(CONST_BG_CLASS)).astype(int)
    c = np.zeros(shape=[ref_map.shape[CONST_ROW_AXIS], ref_map.shape[CONST_COLUMNS_AXIS], max_])
    rows, columns = list(range(ref_map.shape[CONST_ROW_AXIS])), list(range((ref_map.shape[CONST_COLUMNS_AXIS])))
    for i, j in product(rows, columns):
        c[i, j, ref_map[i, j].astype(int)] = CONST_CLASS_LABEL
    return c


def get_guided_image(data):
    return np.mean(data, axis=CONST_SPECTRAL_AXIS)


def construct_new_ref_map(labels, samples, shape):
    labels += CONST_CLASS_LABEL
    new_ref_map = np.zeros(shape=shape)
    for i, indexes in enumerate(samples):
        new_ref_map[indexes[CONST_ROW_AXIS], indexes[CONST_COLUMNS_AXIS]] = labels[i]
    return new_ref_map


def train_svm(data, test_labels, test_samples, train_labels, train_samples):
    model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1024, gamma=2, decision_function_shape='ovo',
                                        probability=True, class_weight='balanced'))
    model.fit(get_data_by_indexes(train_samples, data), train_labels)
    prediction = model.predict(get_data_by_indexes(test_samples, data))
    print('Fitness score: {0:5.2f}%'.format(model.score(get_data_by_indexes(test_samples, data), test_labels) * 100.0))
    return prediction


def generate_pseudo_ground_truth_map(args):
    data, ref_map = load_data(path=args.data_path, ref_map_path=args.ref_map_path)
    guided_image = get_guided_image(data)
    train_samples, train_labels, test_samples, test_labels = prepare_datasets(ref_map=ref_map,
                                                                              training_patch=args.training_patch)

    prediction = train_svm(data, test_labels, test_samples, train_labels, train_samples)

    updated_ref_map = construct_new_ref_map(np.concatenate((train_labels, prediction)),
                                            train_samples + test_samples, ref_map.shape)
    one_hot_ref_map = one_hot_map(updated_ref_map)
    pseudo_ground_truth_map = edge_preserving_filter(ref_map=one_hot_ref_map, neighbourhood_size=args.r,
                                                     guided_image=guided_image)
    np.save(os.path.join(args.dest_path, 'pseudo_ground_truth_map'),
            pseudo_ground_truth_map)


if __name__ == '__main__':
    generate_pseudo_ground_truth_map(arg_parser())
