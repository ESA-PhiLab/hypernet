from random import shuffle

import numpy as np
from python_research.augmentation.transformations import ITransformation
from python_research.experiments.utils.datasets.hyperspectral_dataset import \
    Dataset
from typing import Dict


class Augmenter:
    """
    Class responsible for augmenting the provided dataset. Augmentation process
    is based on the provided transformation.
    """
    def __init__(self, transformation: ITransformation,
                 sampling_mode: str='max_twice'):
        """
        :param transformation: Transformation to be applied to the dataset
        :param sampling_mode: 'max_twice: If twice the number of samples
        for each class does not exceed the number of samples in most numerous class,
        then the count will be doubled, if it does, the number of generated samples
        can be calculated as a difference between most numerous class count and
        number of samples in given class.
        'twice' double the number of samples for each
        class
        """
        self.transformation = transformation
        self.sampling_mode = sampling_mode

    def augment(self, dataset: Dataset, transformations: int=4):
        labels, class_counts = np.unique(dataset.get_labels(), return_counts=True)
        class_counts = dict(zip(labels, class_counts))
        augmented_count = self._calculate_augmented_count_per_class(class_counts)
        indices_to_augment = []
        augmented_labels = []
        for label in augmented_count.keys():
            indices = np.where(dataset.get_labels() == label)[0]
            shuffle(indices)
            indices = indices[:augmented_count[label]]
            indices_to_augment += list(indices)
            augmented_labels += [label for _ in range(augmented_count[label])]
        to_augment = dataset.get_data()[indices_to_augment, ...]
        return self.transformation.transform(to_augment, transformations), \
               np.array(augmented_labels)

    def _calculate_augmented_count_per_class(self, class_counts: Dict[int, int]):
        augmented_count = dict.fromkeys(class_counts.keys())
        if self.sampling_mode == 'max_twice':
            most_numerous_class = max(class_counts.values())
            for label in class_counts.keys():
                if class_counts[label] * 2 < most_numerous_class:
                    augmented_count[label] = class_counts[label]
                else:
                    augmented_count[label] = most_numerous_class - \
                                             class_counts[label]
            return augmented_count
        elif self.sampling_mode == 'twice':
            for label in class_counts.keys():
                augmented_count[label] = class_counts[label]
            return augmented_count
