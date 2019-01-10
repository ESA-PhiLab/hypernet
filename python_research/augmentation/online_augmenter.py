import numpy as np
from collections import OrderedDict

from python_research.experiments.utils.datasets.hyperspectral_dataset import Dataset
from python_research.augmentation.transformations import ITransformation
from keras.models import Model

TOTAL_COUNT = 0
CORRECT_COUNT = 1


class OnlineAugmenter:
    """
    Class for evaluating and predicting class labels using augmentation. Each
    sample is augmented according to the provided Transformation, then each
    augmentation of a given sample is classified by the model as one of the
    classes. Then 'voting' is performed, where class of such a sample is
    concluded as the most occurring class withing those returned by the model.
    """
    def __init__(self):
        self.class_accuracy = None

    def evaluate(self, model: Model, dataset: Dataset,
                 transformation: ITransformation, transformations: int=4):
        """
        Evaluate overall and class accuracy for provided dataset using given
        transformation to augment each sample.
        :param model: Keras model for predicting labels
        :param dataset: Dataset for which the accuracies should be calculated
        :param transformation: Transformation to be used as an augmentation
                               method
        :return: float, np.ndarray: Overall accuracy and a list of accuracies
                                    for each class
        """
        samples_count = len(dataset)
        labels = np.unique(dataset.get_labels())
        class_counts = OrderedDict((label, [0, 0]) for label in labels)
        for sample_index in range(samples_count):
            sample, label = dataset[sample_index]
            augmented_samples = transformation.transform(sample,
                                                         transformations)
            augmented_samples = np.expand_dims(augmented_samples, axis=-1)
            sample = np.expand_dims(sample, axis=-1)
            sample = np.expand_dims(sample, axis=0)
            to_predict = np.vstack([sample, augmented_samples])
            predictions = model.predict(x=to_predict)
            predictions = np.argmax(predictions, axis=1)
            predicted_label = self._vote(predictions)
            class_counts[label][TOTAL_COUNT] += 1
            if predicted_label == label:
                class_counts[label][CORRECT_COUNT] += 1
        accuracy = self._calculate_overall_accuracy(class_counts)
        class_accuracies = self._calculate_class_accuracies(class_counts)
        return accuracy, class_accuracies

    @staticmethod
    def _vote(predictions: np.ndarray):
        unique, counts = np.unique(predictions, return_counts=True)
        max_index = np.argmax(counts)
        return unique[max_index]

    @staticmethod
    def _calculate_class_accuracies(class_counts: dict):
        accuracies = []
        for key in class_counts.keys():
            accuracies.append(float(class_counts[key][CORRECT_COUNT]) /
                              float(class_counts[key][TOTAL_COUNT]))
        return np.array(accuracies)

    @staticmethod
    def _calculate_overall_accuracy(class_counts: dict):
        total_count = 0
        correct_count = 0
        for key in class_counts.keys():
            total_count += class_counts[key][TOTAL_COUNT]
            correct_count += class_counts[key][CORRECT_COUNT]
        return float(correct_count) / float(total_count)
