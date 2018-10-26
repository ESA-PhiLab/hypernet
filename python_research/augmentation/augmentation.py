from typing import Dict, List

import torch
import numpy as np
from random import shuffle
from keras.utils import to_categorical
from scipy.ndimage.filters import gaussian_filter

SMALLEST_POSSIBLE_BATCH_SIZE = 2


def get_samples_per_class_indices(labels: np.ndarray) -> \
        Dict[int, List[int]]:
    """
    Obtain indices of samples for each class
    :param labels: List of class labels
    :return: Dictionary where the key is a class label and value is a
             list with indices
    """
    classes = np.unique(labels)
    samples_per_class = dict.fromkeys(classes)
    for label in classes:
        samples_per_class[label] = np.where(labels == label)[0]
    return samples_per_class


def get_most_numerous_class(samples_per_class: Dict[int, List[int]]) -> int:
    """
    Get count of the most numerous class
    :param samples_per_class: Dictionary with keys as class labels and values
                              as either list of indices or class count itself
    :return: Most numerous class count
    """
    max_ = -np.inf
    for label in samples_per_class:
        if type(samples_per_class[label]) is int:
            if samples_per_class[label] > max_:
                max_ = samples_per_class[label]
        else:
            if len(samples_per_class[label]) > max_:
                max_ = len(samples_per_class[label])
    return max_


def gaussian_filter_augmentation(data: np.ndarray, labels: np.ndarray,
                                 sigma: float=2.0):
    """
    Augment samples using gaussian filter. If twice the number of samples for
    each class does not exceed the number of samples in most numerous class,
    then the count will be doubled, if it does, the number of augmented samples
    can be calculated as a difference between most numerous class count and
    number of samples in given class.
    :param data: Numpy array with samples (channels last).
    :param labels: Array of labels
    :param sigma: Standard deviation parameter for gaussian filter
    :return: (Numpy array with original and augmented samples, Numpy array with
              respective labels)
    """
    augmented_x = []
    augmented_y = []
    samples_per_class = get_samples_per_class_indices(labels)
    most_numerous_class = get_most_numerous_class(samples_per_class)

    for label in samples_per_class:
        to_augment = calculate_how_many_to_augment(label, most_numerous_class,
                                                   samples_per_class)
        shuffle(samples_per_class[label])
        augment_with_gaussian(augmented_x, augmented_y, label,
                              samples_per_class,
                              sigma, to_augment, data)

    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)
    data = np.concatenate([data, augmented_x], axis=0)
    labels = np.concatenate([labels, augmented_y], axis=0)
    return data, labels


def augment_with_gaussian(augmented_x, augmented_y, label,
                          samples_per_class, sigma,
                          to_augment, x_train):
    for augmented_sample in range(to_augment):
        original_sample = x_train[samples_per_class[label][augmented_sample], ...]
        augmented = gaussian_filter(original_sample, sigma)
        augmented_x.append(augmented)
        augmented_y.append([label])


def calculate_stds_of_bands_per_class(x_train, samples_per_class):
    band_stds = dict.fromkeys(samples_per_class.keys())
    for label in samples_per_class:
        band_stds[label] = np.std(x_train[samples_per_class[label]], axis=0)
    return band_stds


def band_noise_augmentation(data, labels, alpha=0.25):
    """
    Calculates standard deviation for each band and for each class, then draws
    a random number from normal distribution with mean=0 and previously
    calculated standard deviation and adds it to every band of a pixel.
    :param data: Numpy array with samples (channels last).
    :param labels: Array of labels
    :param alpha: Parameter used for multiplying the randomly drawn value
    :return: (Numpy array with original and augmented samples, Numpy array with
              respective labels)
    """
    augmented_x = []
    augmented_y = []
    samples_per_class = get_samples_per_class_indices(labels)
    most_numerous_class = get_most_numerous_class(samples_per_class)
    bands_stds = calculate_stds_of_bands_per_class(data, samples_per_class)

    for label in samples_per_class:
        to_augment = calculate_how_many_to_augment(label, most_numerous_class,
                                                   samples_per_class)
        shuffle(samples_per_class[label])
        augment_with_noise(alpha, augmented_x, augmented_y, bands_stds,
                           label, samples_per_class, to_augment, data)

    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)
    data = np.concatenate([data, augmented_x], axis=0)
    labels = np.concatenate([labels, augmented_y], axis=0)
    return data, labels


def augment_with_noise(alpha, augmented_x, augmented_y, bands_stds, label,
                       samples_per_class, to_augment, x_train):
    for sample in range(to_augment):
        augmented = np.empty(x_train.shape[1:])
        for band in range(x_train.shape[-1]):
            noise = alpha * np.random.normal(loc=0, scale=bands_stds[label][band])
            augmented[band] = x_train[samples_per_class[label][sample]][band] + noise
        augmented_x.append(augmented)
        augmented_y.append([label])


def calculate_how_many_to_augment(label, most_numerous_class, samples_per_class,
                                  mode):
    if type(samples_per_class[label]) is int:
        class_count = samples_per_class[label]
    else:
        class_count = len(samples_per_class[label])
    if mode == 'balanced_full':
        to_augment = class_count
    if mode == 'balanced_half':
        to_augment = int(class_count * 0.5)
    elif mode == 'unbalanced':
        if class_count * 2 >= most_numerous_class:
            to_augment = most_numerous_class - class_count
        else:
            to_augment = class_count
    return to_augment


def generate_samples(generator: torch.nn.Module,
                     samples_per_class: Dict[int, List[int]],
                     bands_count: int,
                     classes_count: int,
                     device='cpu',
                     augmentation_mode: str='balanced'):
    """
    Generate samples using provided generator. If twice the number of samples
    for each class does not exceed the number of samples in most numerous class,
    then the count will be doubled, if it does, the number of generated samples
    can be calculated as a difference between most numerous class count and
    number of samples in given class.
    :param generator: Already trained Generator object
    :param samples_per_class: Dictionary with keys as class labels and values
                              as either list of indices or class count itself
    :param bands_count: Number of bands in the dataset
    :param classes_count: Number of classes in the dataset
    :param device: either 'gpu' or 'cpu'
    :param augmentation_mode: balanced_full, balanced_half or unbalanced.
                              balanced_full: all classes count will be doubled.
                              balanced_half: all classes count will be
                                             increased by half of the current
                                             size.
                              unbalanced: If twice the number of samples
    for each class does not exceed the number of samples in most numerous class,
    then the count will be doubled, if it does, the number of generated samples
    can be calculated as a difference between most numerous class count and
    number of samples in given class.
    :return: (Tensor with generated samples, Tensor with respective labels)
    """

    most_numerous_class = get_most_numerous_class(samples_per_class)
    generated_x = torch.Tensor(0)
    if device == 'gpu':
        generated_x = generated_x.cuda()
    generated_y = []
    for label in samples_per_class:
        to_augment = calculate_how_many_to_augment(label, most_numerous_class,
                                                   samples_per_class,
                                                   augmentation_mode)
        if to_augment < SMALLEST_POSSIBLE_BATCH_SIZE:
            continue
        noise = torch.FloatTensor(np.random.normal(0.5, 0.1, (to_augment,
                                                              bands_count)))
        label_one_hot = to_categorical(np.full(to_augment, label),
                                       classes_count)
        label_one_hot = torch.from_numpy(label_one_hot)
        if device == 'gpu':
            noise = noise.cuda()
            label_one_hot = label_one_hot.cuda()
        generated = generator(noise, label_one_hot)
        generated_x = torch.cat([generated_x, generated])
        generated_y += [label for _ in range(to_augment)]
    return generated_x, generated_y
