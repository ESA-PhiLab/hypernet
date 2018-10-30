"""
Helper procedures for jupyter notebook
"""
from random import randint
from base64 import b64encode
from io import BytesIO
import imageio
import os
from python_research.experiments.utils.datasets.hyperspectral_dataset import HyperspectralDataset
from python_research.experiments.utils.datasets.concat_dataset import ConcatDataset
from ipyleaflet import Map, ImageOverlay
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix
import re


def normalize_to_zero_one(image_data: np.ndarray) -> np.ndarray:
    """
    Normalizes image data to zero-one floating point range
    :param image_data: Image data to normalize
    :return: Normalized image data
    """
    max_value = image_data.max()
    if max_value == 0:
        return image_data.astype(np.float32)

    return image_data.astype(np.float32) / image_data.max()


def normalize_to_byte(image_data: np.ndarray) -> np.ndarray:
    """
    Normalizes image data to 0-255 8-bit integer range
    :param image_data: Image data to normalize
    :return: Normalized image data
    """
    byte_data = 255 * normalize_to_zero_one(image_data)

    return byte_data.astype(np.uint8)


def serialize_to_url(image_data: np.ndarray) -> str:
    """
    Serializes image data to base64 encoded png format
    :param image_data: Image data to serialize
    :return: String containing serialized image data
    """
    in_memory_file = BytesIO()

    imageio.imwrite(in_memory_file, image_data, format='png')

    ascii_data = b64encode(in_memory_file.getvalue()).decode('ascii')

    return 'data:image/png;base64,' + ascii_data


def create_map(normalized_image: np.ndarray) -> Map:
    """
    Creates leaflet map with given image
    :param normalized_image: Image data normalized to 0-255 8-bit integer
    :return: Leaflet map
    """
    width = normalized_image.shape[0]
    height = normalized_image.shape[1]
    bounds = [(-width / 2, -height / 2), (width / 2, height / 2)]

    layer = ImageOverlay(url=serialize_to_url(normalized_image), bounds=bounds)
    leaflet = Map(center=[0, 0], zoom=1, interpolation='nearest')
    leaflet.clear_layers()
    leaflet.add_layer(layer)

    return leaflet


def create_image(normalized_image: np.ndarray, label: str=None):
    """
    Creates jupyter image with given image data
    :param normalized_image: Image data normalized to 0-255 8-bit integer
    :return: Jupyter image
    """
    plt.figure(figsize=(10, 10))

    if label is not None:
        plt.title(label)

    if normalized_image.shape[2] == 1:
        plt.imshow(np.repeat(normalized_image, 3, axis=2), interpolation='none')
    else:
        plt.imshow(normalized_image, interpolation='none')


def show_samples_location(dataset, neighbourhood, samples_to_show_count):
    class_to_display = randint(0, len(np.unique(dataset.y)))
    train_indices = dataset.train_indices[class_to_display][0:samples_to_show_count]
    test_indices = dataset.test_indices[class_to_display][0:samples_to_show_count]
    im = dataset.x[:, :, randint(0, dataset.x.shape[-1])]
    fig, ax = plt.subplots(1)
    ax.imshow(im)
    for train in train_indices:
        x = [train.y - int(neighbourhood[0]/2), train.x - int(neighbourhood[1]/2)]
        ax.add_patch(Rectangle(x, neighbourhood[0], neighbourhood[1], color='r', fill=False))

    for test in test_indices:
        x = [test.y - int(neighbourhood[0]/2), test.x - int(neighbourhood[1]/2)]
        ax.add_patch(Rectangle(x, neighbourhood[0], neighbourhood[1], color='y', fill=False))
    plt.show()


def calculate_class_accuracy(y_pred: np.ndarray,
                             y_true: np.ndarray,
                             classes_count: int) -> np.ndarray:
    """
    Calculate per class accuracy for given predictions
    :param y_pred: Predictions provided as a list with a predicted class number
    :param y_true: True value of a class
    :param classes_count: Number of classes in the dataset
    :return: An accuracy for each class individually
    """
    matrix = confusion_matrix(y_true, y_pred,
                              labels=[x for x in range(classes_count)])
    matrix = matrix / matrix.astype(np.float32).sum(axis=1)
    return np.diagonal(matrix)


def load_patches(directory: os.PathLike, neighbourhood_size: int=1):
    patches_paths = [x for x in os.listdir(directory)
                     if 'gt' not in x and 'patch' in x]
    gt_paths = [x for x in os.listdir(directory) if 'gt' in x and 'patch' in x]
    test_paths = [x for x in os.listdir(directory) if 'test' in x and '.npy' in x]
    patches_paths = sorted_alphanumeric(patches_paths)
    gt_paths = sorted_alphanumeric(gt_paths)
    data = []
    for patch_path, gt_path in zip(patches_paths, gt_paths):
        data.append(HyperspectralDataset(os.path.join(directory, patch_path),
                                         os.path.join(directory, gt_path),
                                         neighbourhood_size))
    test_data = HyperspectralDataset(os.path.join(directory, test_paths[0]),
                                     os.path.join(directory, test_paths[1]),
                                     neighbourhood_size)
    return ConcatDataset(data), test_data


def combine_patches(patches, patches_gt, test, test_gt, neighbourhood_size:int=1):
    data = []
    for patch, gt in zip(patches, patches_gt):
        data.append(HyperspectralDataset(patch, gt, neighbourhood_size))
    test_set = HyperspectralDataset(test, test_gt, neighbourhood_size)
    return ConcatDataset(data), test_set


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)
