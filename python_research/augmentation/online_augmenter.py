import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple

from python_research.dataset_structures import Dataset
from python_research.augmentation.transformations import ITransformation
from keras.models import Model

TOTAL_COUNT = 0
CORRECT_COUNT = 1


class OnlineAugmenter:
    """
    Class for evaluating and predicting class labels using online
    augmentation. Each sample is augmented according to the provided
    Transformation, then each augmentation of a given sample is classified by
    the model as one of the classes. Afterwards, 'voting' is performed where
    the final prediction is concluded as the most occurring class withing those
    returned by the model.
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
        :param transformations: Number of transformations to perform on each
                                sample
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

    def predict(self, model: Model, dataset: Dataset,
                transformation: ITransformation, transformations: int=4) -> \
            List[int]:
        """
        Predict labels for given dataset
        :param model: Keras model for predicting labels
        :param dataset: Dataset for which the labels should be predicted
        :param transformation: Transformation t obe used as an augmentation
                               method
        :param transformations: Number of transformations to perform on each
                                sample
        :return: List with predicted labels
        """
        samples_count = len(dataset)
        predicted_labels = []
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
            predicted_labels.append(predicted_label)
        return predicted_labels

    @staticmethod
    def _vote(predictions: np.ndarray) -> int:
        """
        Perform the majority voting
        :param predictions: Predictions returned by the model
        :return: Final label
        """
        unique, counts = np.unique(predictions, return_counts=True)
        max_index = np.argmax(counts)
        return unique[max_index]

    @staticmethod
    def _calculate_class_accuracies(class_counts: Dict[int, Tuple]) \
            -> List:
        """
        Calculate accuracy for each class separately
        :param class_counts: Dictionary with the number of correctly predicted
        samples as well as with total number of samples for each class.
        :return: List of accuracies for each class
        """
        accuracies = []
        for key in class_counts.keys():
            accuracies.append(float(class_counts[key][CORRECT_COUNT]) /
                              float(class_counts[key][TOTAL_COUNT]))
        return np.array(accuracies)

    @staticmethod
    def _calculate_overall_accuracy(class_counts: Dict[int, Tuple]) \
            -> int:
        """
        Calculate overall accuracy (percentage of correctly predicted samples)
        :param class_counts: Dictionary with the number of correctly predicted
        samples as well as with total number of samples for each class.
        :return: Percentage of correctly predicted samples
        """
        total_count = 0
        correct_count = 0
        for key in class_counts.keys():
            total_count += class_counts[key][TOTAL_COUNT]
            correct_count += class_counts[key][CORRECT_COUNT]
        return float(correct_count) / float(total_count)
