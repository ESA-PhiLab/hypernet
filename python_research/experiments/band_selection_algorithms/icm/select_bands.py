import argparse
import os

from sklearn import svm

from python_research.experiments.band_selection_algorithms.icm.improved_class_map import prepare_datasets, \
    get_data_by_indexes
from python_research.experiments.band_selection_algorithms.utils import *


def train_classifiers(br: list, bs: np.ndarray, train_samples: list, train_labels: list,
                      test_samples: list, test_labels: list) -> list:
    """
    Train SVMs on each band and obtain accuracies.

    :param br: Set of bands.
    :param bs: Set of selected bands.
    :param train_samples: Indexed training data.
    :param train_labels: Indexed training labels.
    :param test_samples: Indexed test data.
    :param test_labels: Indexed test labels.
    :return: List of accuracy scores for all combined bands.
    """
    band_scores = []
    model = svm.SVC(kernel="rbf", C=1024, gamma=2, decision_function_shape="ovo",
                    probability=True, class_weight="balanced")
    if bs is not None:
        for i in range(br.__len__()):
            if br[i] is not None:
                if bs.shape.__len__() == 2:
                    bs = np.expand_dims(bs, axis=SPECTRAL_AXIS)
                br[i] = np.concatenate((np.expand_dims(br[i], axis=SPECTRAL_AXIS),
                                        bs), axis=SPECTRAL_AXIS)

    for i in range(br.__len__()):
        if br[i] is not None:
            print("Band index: {}".format(str(i)))
            if bs is None:
                train_data, test_data = np.expand_dims(get_data_by_indexes(train_samples, br[i]), axis=SPECTRAL_AXIS), \
                                        np.expand_dims(get_data_by_indexes(test_samples, br[i]), axis=SPECTRAL_AXIS)
            else:
                print("Combined band depth: {}".format(br[i].shape[SPECTRAL_AXIS]))
                train_data, test_data = get_data_by_indexes(train_samples, br[i]), \
                                        get_data_by_indexes(test_samples, br[i])
            model.fit(train_data, train_labels)
            score = model.score(test_data, test_labels)
            band_scores.append(score)
            print("Score: {}".format(score))
        else:
            band_scores.append(-1)
    return band_scores


def select_bands(args: argparse.Namespace, improved_classification_map: np.ndarray = None):
    """
    Select, save and show selected bands using ICM algorithm.

    :param improved_classification_map: Reference map.
    :param args: Arguments passed.
    """
    if improved_classification_map is None:
        improved_classification_map = np.load(os.path.join(args.dest_path,
                                                           "improved_classification_map_{}.npy".format(
                                                               str(args.bands_num))))
    selected_bands = []
    data = load_data(data_path=args.data_path, ref_map_path=args.ref_map_path)[0]
    data = min_max_normalize_data(data=data)
    bs, br = None, [data[..., i] for i in range(data.shape[SPECTRAL_AXIS])]
    train_samples, train_labels, test_samples, test_labels = prepare_datasets(improved_classification_map,
                                                                              args.training_patch)
    while selected_bands.__len__() < args.bands_num:
        band_scores = train_classifiers(br.copy(), bs, train_samples,
                                        train_labels, test_samples, test_labels)
        band_id = np.argmax(band_scores).astype(int)
        selected_bands.append(band_id)
        if bs is None:
            bs = br[band_id]
        else:
            if bs.shape.__len__() == 2:
                bs = np.expand_dims(bs, axis=SPECTRAL_AXIS)
            bs = np.concatenate((np.expand_dims(br[band_id], axis=SPECTRAL_AXIS), bs), axis=SPECTRAL_AXIS)
        br[band_id] = None

    np.savetxt(fname=os.path.join(args.dest_path, "selected_bands_{}".format(str(args.bands_num))),
               X=np.sort(np.asarray(selected_bands)), fmt="%d")
    print("Selected bands: {}".format(np.sort(np.asarray(selected_bands))))
    print("Number of selected bands: {}".format(np.unique(selected_bands).size))
