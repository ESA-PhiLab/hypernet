import argparse
import numpy as np

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def change_data_formatting(x, y):
    x = x.reshape(x.shape[0], x.shape[-1])
    y = y.reshape(y.shape[0], y.shape[-1])
    y = np.argmax(y, axis=1)
    return x.astype(np.float32), y


def load_classifier(classifier):
    if classifier == "svm":
        return SVC
    elif classifier == "dt":
        return DecisionTreeClassifier
    elif classifier == "rf":
        return RandomForestClassifier
    else:
        raise ValueError("Unknown classifier")


def get_parameters(classifier):
    if classifier == "svm":
        return {"C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 500.0],
                "gamma": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 500.0]}
    elif classifier == "dt":
        return {"min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "min_samples_split": [2, 3, 4, 5, 6]}
    elif classifier == "rf":
        return {"n_estimators": [10, 20, 40, 80, 120, 160, 200, 240],
                "min_samples_split": [2, 3, 4, 5, 6]}
    else:
        raise ValueError("Unknown classifier")


def get_accuracy_per_class(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    unique, counts = np.unique(y_true, return_counts=True)
    return np.diag(conf_matrix).astype(np.float32)/counts


def run_crossvalidation(classifier, parameters, x_train, y_train, x_val, y_val, ):
    param_names = list(parameters.keys())
    parameter_space = list()
    for value0 in parameters[param_names[0]]:
        for value1 in parameters[param_names[1]]:
            parameter_space.append({param_names[0]: value0,
                                    param_names[1]: value1})

    best_score = 0
    best_clf = None
    for item in parameter_space:
        try:
            clf = classifier(n_jobs=4, **item)
        except TypeError:
            clf = classifier(**item)
        clf.fit(x_train, y_train)
        score = clf.score(x_val, y_val)
        if best_score > score:
            continue
        else:
            best_score = score
            best_clf = clf

    return best_clf

def arguments():
    parser = argparse.ArgumentParser(description='Input  arguments.')

    parser.add_argument('--dataset',
                        action="store",
                        dest="dataset",
                        type=str,
                        help='Dataset')

    parser.add_argument('--runs',
                        action="store",
                        dest="runs",
                        type=int,
                        help='Number of runs',
                        default=1)

    parser.add_argument('--validation',
                        action="store",
                        dest="validation_proportion",
                        type=float,
                        help='Proportion of validation samples')

    parser.add_argument('--test',
                        action="store",
                        dest="test_proportion",
                        type=float,
                        help='Proportion of test samples')

    parser.add_argument('--version',
                        action="store",
                        dest="version",
                        type=float,
                        help='Version of reduced dataset')

    parser.add_argument('--mode',
                        action="store",
                        dest="mode",
                        type=str,
                        help='Mode of band selection for reduced dataset')

    parser.add_argument('--clf',
                        action="store",
                        dest="classifier",
                        type=str,
                        help='Type of classifier to train')

    return parser.parse_args()