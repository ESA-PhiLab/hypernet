from random import shuffle

import numpy as np
from python_research.augmentation.transformations import ITransformation
from python_research.dataset_structures import \
    Dataset
from utils import calculate_augmented_count_per_class


class OfflineAugmenter:
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
        'twice': double the number of samples for each class
        """
        self.transformation = transformation
        self.sampling_mode = sampling_mode

    def augment(self, dataset: Dataset, transformations: int=1)-> \
            [np.ndarray, np.ndarray]:
        """
        Perform the augmentation
        :param dataset: Dataset to augment
        :param transformations: Number of transformations to perform on each
        sample
        :return: Augmented samples as well as their respective labels
        """
        labels, class_counts = np.unique(dataset.get_labels(), return_counts=True)
        class_counts = dict(zip(labels, class_counts))
        augmented_count = calculate_augmented_count_per_class(class_counts,
                                                              self.sampling_mode)
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
