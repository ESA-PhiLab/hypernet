import torch
import numpy as np
from keras.utils import to_categorical
from typing import List

from python_research.augmentation.GAN.generator import Generator
from python_research.dataset_structures import \
    Dataset
from utils import calculate_augmented_count_per_class

SMALLEST_POSSIBLE_BATCH_SIZE = 2


class SamplesGenerator:
    """
    Class responsible for synthesizing new samples using provided generator
    """
    def __init__(self, device='cpu',
                 noise_mean: float=0.5,
                 noise_std: float=0.1):
        """
        :param device: Whether to use 'cpu' or 'gpu'
        :param noise_mean: center of the random noise provided to the generator
        :param noise_std: standard deviation of the random noise provided
        to the generator
        """
        self.device = device
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def generate(self, dataset: Dataset, generator: Generator,
                 sampling_mode='twice') -> [torch.Tensor, List[int]]:
        """
        Generate samples using provided dataset
        :param dataset: Dataset for which the samples should be synthesized
        :param generator: Already trained generator
        :param sampling_mode: 'twice': Double the number of samples in each class
        'max_twice': If twice the number of samples
        for each class does not exceed the number of samples in most numerous
        class, then the count will be doubled, if it does, the number of
        generated samples will be calculated as a difference between most
        numerous class count and number of samples in given class.
        :return: Synthesized samples (Tensor) and labels of all synthesized
        samples (list)
        """
        labels, class_counts = np.unique(dataset.get_labels(), return_counts=True)
        class_counts = dict(zip(labels, class_counts))
        to_generate_count = calculate_augmented_count_per_class(class_counts,
                                                                sampling_mode)
        generated_x = torch.Tensor(0)
        if self.device == 'gpu':
            generated_x = generated_x.cuda()
        generated_y = []
        for label in to_generate_count.keys():
            to_generate = to_generate_count[label]
            if to_generate < SMALLEST_POSSIBLE_BATCH_SIZE:
                continue
            noise = torch.FloatTensor(np.random.normal(self.noise_mean,
                                                       self.noise_std,
                                                       (to_generate,
                                                        dataset.shape[-1])))
            label_one_hot = to_categorical(np.full(to_generate, label),
                                           len(class_counts))
            label_one_hot = torch.from_numpy(label_one_hot)
            if self.device == 'gpu':
                noise = noise.cuda()
                label_one_hot = label_one_hot.cuda()
            generated = generator(noise, label_one_hot)
            generated_x = torch.cat([generated_x, generated])
            generated_y += [label for _ in range(to_generate)]
        return generated_x, generated_y
