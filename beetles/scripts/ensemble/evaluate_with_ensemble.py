"""
Evaluate the dataset using an ensemble of models.
"""

import clize
import numpy as np
from sklearn.metrics import confusion_matrix

from ml_intuition import enums
from ml_intuition.data import io
from ml_intuition.evaluation.performance_metrics import get_model_metrics, \
    get_fair_model_metrics
from ml_intuition.evaluation.time_metrics import timeit
from ml_intuition.models import Ensemble


def evaluate(*,
             y_pred: np.ndarray,
             data,
             dest_path: str,
             n_classes: int,
             model_path: str,
             voting: str = 'hard',
             train_set_predictions: np.ndarray = None,
             voting_model: str = None,
             voting_model_params: str = None):
    """
    Function for evaluating the trained model.

    :param y_pred: Predictions of test set.
    :param model_path: Path to the model.
    :param data: Either path to the input data or the data dict.
    :param dest_path: Directory in which to store the calculated metrics
    :param n_classes: Number of classes.
    :param voting: Method of ensemble voting. If ‘hard’, uses predicted class
        labels for majority rule voting. Else if ‘soft’, predicts the class
        label based on the argmax of the sums of the predicted probabilities.
        Else if 'booster', employs a new model, which is trained on the
        ensemble predictions on the training set.
    :param train_set_predictions: Predictions of the train set. Only used if
        'voting' = 'classifier'.
    :param voting_model: Type of model to use when the voting argument is set
        to "booster". This indicates, that a new model is trained on the
        ensemble predictions on the learning set,
        to leverage the quality of the classification or
        regression. Supported models are: SVR, SVC (support vector machine for
        regression and classification), RFR, RFC (random forest for regression
        and classification), DTR, DTC (decision tree for regression and
        classification).
    :param voting_model_params: Parameters of the voting model,
        should be specified in the same manner
        as the parameters for the noise injection.
    """

    ensemble = Ensemble(voting=voting)
    if voting == 'booster':
        train_set_predictions = np.array(train_set_predictions)
        ensemble.train_ensemble_predictor(
            train_set_predictions,
            data[enums.Dataset.TRAIN][enums.Dataset.LABELS],
            voting_model,
            voting_model_params)
    vote = timeit(ensemble.vote)
    y_pred, voting_time = vote(y_pred)

    y_true = data[enums.Dataset.TEST][enums.Dataset.LABELS]
    model_metrics = get_model_metrics(y_true, y_pred)
    model_metrics['inference_time'] = [voting_time]
    conf_matrix = confusion_matrix(y_true, y_pred,
                                   labels=[i for i in range(n_classes)])
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
    io.save_confusion_matrix(conf_matrix, dest_path)
    if enums.Splits.GRIDS in model_path:
        if type(data) is str:
            train_dict = io.extract_set(data, enums.Dataset.TRAIN)
            labels_in_train = np.unique(train_dict[enums.Dataset.LABELS])
        else:
            train_labels = data[enums.Dataset.TRAIN][enums.Dataset.LABELS]
            if train_labels.ndim > 1:
                train_labels = np.argmax(train_labels, axis=-1)
            labels_in_train = np.unique(train_labels)
        fair_metrics = get_fair_model_metrics(conf_matrix, labels_in_train)
        io.save_metrics(dest_path=dest_path,
                        file_name=enums.Experiment.INFERENCE_FAIR_METRICS,
                        metrics=fair_metrics)


if __name__ == '__main__':
    clize.run(evaluate)
