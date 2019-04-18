import argparse
import os

from sklearn import svm

from python_research.experiments.band_selection_algorithms.BS_IC.improved_class_map import prepare_datasets, \
    get_data_by_indexes
from python_research.experiments.band_selection_algorithms.utils import *


def train_classifiers(br, train_samples: list, train_labels: list,
                      test_samples: list, test_labels: list, bs=None):
    """
    Train SVMs on each band and obtain accuracies.

    :param br: Set of bands.
    :param train_samples: Indexed training data.
    :param train_labels: Indexed training labels.
    :param test_samples: Indexed test data.
    :param test_labels: Indexed test labels.
    :param bs: Set of selected bands.
    :return:
    """
    band_scores = []
    model = svm.SVC(kernel='rbf', C=1024, gamma=2, decision_function_shape='ovo',
                    probability=True, class_weight='balanced')
    if bs is not None:
        temp_br = []
        for i in range(br.shape[SPECTRAL_AXIS]):
            temp_br.append(np.concatenate((np.expand_dims(br[..., i], -1), bs), axis=-1))
        for i in range(temp_br.__len__()):
            print('Band index: {}'.format(str(i)))
            model.fit(get_data_by_indexes(train_samples, temp_br[i]), train_labels)
            score = model.score(get_data_by_indexes(test_samples, temp_br[i]), test_labels)
            print('Score: ', score)
            band_scores.append(score)
    else:
        temp_br = br.copy()
        for i in range(temp_br.shape[SPECTRAL_AXIS]):
            print('Band index: {}'.format(str(i)))
            model.fit(np.expand_dims(get_data_by_indexes(train_samples, temp_br[..., i]), axis=-1), train_labels)
            score = model.score(np.expand_dims(get_data_by_indexes(test_samples, temp_br[..., i]), axis=-1),
                                test_labels)
            band_scores.append(score)
            print('Score: ', score)
    return band_scores


def select_bands(pseudoground_truth_map: np.ndarray, args: argparse.Namespace):
    """
    Select, save and show selected bands.

    :param pseudoground_truth_map: Reference map.
    :param args: Arguments passed.
    """
    selected_bands = []
    data = load_data(data_path=args.data_path, ref_map_path=args.ref_map_path, get_ref_map=False)
    bs, br = [], data
    train_samples, train_labels, test_samples, test_labels = prepare_datasets(pseudoground_truth_map,
                                                                              args.training_patch)
    while selected_bands.__len__() < args.bands_num:
        if selected_bands.__len__() == 0:
            band_scores = train_classifiers(br, train_samples, train_labels, test_samples, test_labels)
        else:
            band_scores = train_classifiers(br, train_samples, train_labels, test_samples, test_labels,
                                            np.moveaxis(bs, 0, -1))
        for i in selected_bands:
            band_scores[i] = -1
        band_id = np.argmax(band_scores).astype(int)
        selected_bands.append(band_id)
        bs.append(br[..., band_id])
        print('Len: ', selected_bands.__len__())

    np.savetxt(fname=os.path.join(args.dest_path, "selected_bands_{}".format(str(args.bands_num))),
               X=np.sort(np.asarray(selected_bands)), fmt="%d")
    print("Selected bands: {}".format(np.sort(np.asarray(selected_bands))))
