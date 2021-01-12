from math import ceil
from typing import Tuple, Dict

import pandas as pd
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneOut

from ml_intuition.models import ML_MODELS, ML_MODELS_GRID


def train_and_eval_regression(dataframe: pd.DataFrame,
                              label_name: str,
                              train_fraction: float,
                              model_name: str,
                              seed: int,
                              verbose: int = 1,
                              n_jobs: int = 4) -> Tuple[pd.DataFrame, Dict]:
    """
    Train and evaluate regression given dataset as a dataframe.
    The dataset is a design matrix, where in rows each new observations
    are placed, and the columns denote the explanatory variables.
    The process of finding the best parameters is done by leave one
    out cross validation method, utilizing the mean-squared error.

    :param dataframe: Data collected for the regression problem.
    :param label_name: Name of the label i.e., the dependent variable.
    :param train_fraction: Fraction of samples for sets sampling.
    :param model_name: Name of the utilized model.
    :param seed: Seed used for reproduction of experiment results.
    :param verbose: Verbosity mode.
    :param n_jobs: Number of jobs for parallel computing.
    :return: Tuple of the report over the test set as a dataframe
        and the best parameters found as a dictionary.
    """
    train = dataframe.sample(n=ceil(train_fraction * len(dataframe)),
                             random_state=seed)
    test = dataframe.drop(train.index)

    X_train, y_train, X_test, y_test = \
        train.drop(columns=label_name), train[label_name], \
        test.drop(columns=label_name), test[label_name]

    model = GridSearchCV(
        estimator=ML_MODELS[model_name](random_state=seed),
        param_grid=ML_MODELS_GRID[model_name],
        cv=LeaveOneOut().split(X_train, y_train),
        scoring=make_scorer(mean_squared_error),
        verbose=verbose,
        n_jobs=n_jobs,
        refit=True).fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    return pd.DataFrame.from_dict({'r2': [r2],
                                   'mse': [mse]}), model.best_params_
