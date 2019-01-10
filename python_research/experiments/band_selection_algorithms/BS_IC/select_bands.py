import os

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from python_research.experiments.band_selection_algorithms.BS_IC.improved_class_map import prepare_datasets, \
    get_data_by_indexes
from python_research.experiments.band_selection_algorithms.BS_IC.utils import *


def train_classifiers(data, train_samples, train_labels, test_samples, test_labels):
    """
    Train SVMs on each band and obtain accuracies.

    :param data: Data block containing all bands.
    :param train_samples: Indexed training data.
    :param train_labels: Indexed training labels.
    :param test_samples: Indexed test data.
    :param test_labels: Indexed test labels.
    :return:
    """
    band_scores = []
    for i in range(data.shape[CONST_SPECTRAL_AXIS]):
        model = OneVsRestClassifier(svm.SVC(kernel='rbf', C=1024, gamma=2, decision_function_shape='ovo',
                                            probability=True, class_weight='balanced'))
        model.fit(np.expand_dims(get_data_by_indexes(train_samples, data[..., i]), axis=-1), train_labels)
        band_scores.append(
            model.score(np.expand_dims(get_data_by_indexes(test_samples, data[..., i]), axis=-1), test_labels))
    return band_scores


def select_bands(ref_map, args):
    """
    Select args.bands_num bands based on SVM fitness scores.

    :param ref_map: Reference map.
    :param args: Arguments passed.
    """
    selected_bands = []
    data = load_data(path=args.data_path, ref_map_path=args.ref_map_path, get_ref_map=False)
    train_samples, train_labels, test_samples, test_labels = prepare_datasets(ref_map, args.training_patch)
    band_scores = train_classifiers(data, train_samples, train_labels, test_samples, test_labels)
    print('Band SVM scores: {}'.format(band_scores))
    while selected_bands.__len__() < args.bands_num:
        band_id = np.argmax(band_scores).astype(int)
        selected_bands.append(band_id)
        band_scores[band_id] = SELECTED_BAND_FLAG
    np.savetxt(fname=os.path.join(args.dest_path, 'selected_bands'),
               X=np.sort(np.asarray(selected_bands)), fmt='%d')
    print('Selected bands: {}'.format(np.sort(np.asarray(selected_bands))))


if __name__ == '__main__':
    args = arg_parser()
    map_ = np.load(os.path.join(args.dest_path, 'pseudo_ground_truth_map.npy')).astype(int)
    select_bands(map_, args)
