from math import ceil
from typing import Tuple, Dict

import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, \
    balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from ml_intuition.models import ML_MODELS, ML_MODELS_GRID


def train_and_eval_classifier(dataframe: pd.DataFrame,
                              label_name: str,
                              train_fraction: float,
                              model_name: str,
                              seed: int,
                              verbose: int = 1,
                              n_jobs: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """
    Train and evaluate the classifier given dataset as a dataframe.
    The dataset is a design matrix, where in rows each new observations
    are placed, and the columns denote the explanatory variables.
    The process of finding the best parameters is done by leave one
    out cross validation method, utilizing the accuracy score.

    :param dataframe: Data collected for the classification problem.
    :param label_name: Name of the label i.e., the dependent variable.
    :param train_fraction: Fraction of samples for
        each class for stratified sampling.
    :param model_name: Name of the utilized model.
    :param seed: Seed used for reproduction of experiment results.
    :param verbose: Verbosity mode.
    :param n_jobs: Number of jobs utilized for the parallel computing.
    :return: Tuple of the report over the test set as a dataframe
        and the best parameters found as a dictionary.
    """
    dataframe = dataframe.join(
        pd.get_dummies(dataframe[label_name], prefix='class'))
    class_names = [col_name for col_name in dataframe
                   if col_name.startswith('class')]

    train = dataframe.groupby('label', group_keys=False).apply(
        lambda class_group: class_group.sample(
            n=ceil(train_fraction * len(class_group)),
            random_state=seed)).drop(columns=label_name)

    test = dataframe.drop(train.index).drop(columns=label_name)

    X_train, y_train, X_test, y_test = \
        train.drop(columns=class_names), train[class_names], \
        test.drop(columns=class_names), test[class_names]

    model = GridSearchCV(
        estimator=ML_MODELS[model_name](random_state=seed),
        param_grid=ML_MODELS_GRID[model_name],
        cv=LeaveOneOut().split(X_train, y_train),
        scoring=make_scorer(accuracy_score),
        verbose=verbose,
        n_jobs=n_jobs,
        refit=True).fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    y_test_true_argmax = y_test.values.argmax(axis=1)
    y_test_pred_argmax = y_test_pred.argmax(axis=1)
    class_names = {class_name: i for i, class_name in enumerate(list(y_test))}

    test_report = pd.DataFrame(
        confusion_matrix(y_true=y_test_true_argmax,
                         y_pred=y_test_pred_argmax,
                         labels=list(class_names.values())),
        index=['true_' + class_name for class_name in class_names.keys()],
        columns=['pred_' + class_name for class_name in class_names.keys()])

    placeholder = [None for _ in range(len(class_names) - 1)]
    test_report['test_oa_acc'] = [accuracy_score(
        y_true=y_test_true_argmax, y_pred=y_test_pred_argmax)] + placeholder
    test_report['test_avg_acc'] = [balanced_accuracy_score(
        y_true=y_test_true_argmax, y_pred=y_test_pred_argmax)] + placeholder
    return test_report, model.best_params_
