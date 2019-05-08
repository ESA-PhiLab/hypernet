from typing import Tuple
import numpy as np
from random import shuffle, randint


class WindowSize(object):
    def __init__(self, x: int, y: int):
        if not x > 0 or not y > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x, y))
        elif not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(type(x),
                                                                      type(y)))
        self.x = x
        self.y = y


class Stride(object):
    def __init__(self, x_stride: int, y_stride: int):
        if not x_stride > 0 or not y_stride > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x_stride,
                                                                  y_stride))
        elif not isinstance(x_stride, int) or not isinstance(y_stride, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(
                    type(x_stride),
                    type(y_stride)))
        self.x = x_stride
        self.y = y_stride


class Patch:
    def __init__(self,
                 index: int,
                 left_x: int,
                 right_x: int,
                 upper_y: int,
                 lower_y: int):
        self.index = index
        self.left_x = left_x
        self.right_x = right_x
        self.upper_y = upper_y
        self.lower_y = lower_y


def sliding_window(image: np.ndarray, window_size: WindowSize,
                   stride: Stride = 0):
    """
    Apply sliding window to the image, which extract patches of the size provided
    as a window_size argument. If the stride argument is not calculated properly,
    there might be some loss of data. The patch has always the same number of
    channels as the original image.
    :param image: Pixels of the image you would like to apply sliding window to
    :param window_size: Size of the sliding window
    :param stride: The amount by which the window should be moved. If not provided,
    the stride will be equal to the window size so no there will be no overlap
    :return: A list of patches
    """
    if not isinstance(window_size, WindowSize):
        raise TypeError("window_size should be a WindowSize class instance")
    if not stride == 0 and not isinstance(stride, Stride):
        raise TypeError("stride should be a Stride class instance")
    if stride == 0:
        stride = Stride(window_size.x, window_size.y)

    number_of_patches_in_x = int(((image.shape[1] - window_size.x) / stride.x) + 1)
    number_of_patches_in_y = int(((image.shape[0] - window_size.y) / stride.y) + 1)
    patches = []
    index = 0
    for y_dim_patch_number in range(0, number_of_patches_in_y):
        for x_dim_patch_number in range(0, number_of_patches_in_x):
            left_border_x = int(0 + stride.x * x_dim_patch_number)
            right_border_x = int(window_size.x + stride.x * x_dim_patch_number)
            upper_border_y = int(0 + stride.y * y_dim_patch_number)
            lower_border_y = int(window_size.y + stride.y * y_dim_patch_number)
            patch = Patch(index, left_border_x, right_border_x,
                          upper_border_y, lower_border_y)
            patches.append(patch)
            index += 1
    return patches


def extract_grids(dataset_path: str, ground_truth_path: str, window_size: Tuple,
                  total_samples_count: int):
    """
    Divide an image into patches and extract them randomly until a provided
    number of samples is contained in those patches. If a patch is empty
    (does not contain any samples, only background) then it is omitted.
    :param dataset_path: Path to the dataset (.npy or .mat format)
    :param ground_truth_path: Path to the ground truth file (.npy or .mat format)
    :param window_size: Size of a single patch
    :param total_samples_count: Desired number of samples contained in
                                all of the extracted patches
    :returns: tuple (extracted_patches, extracted_patches_gt),
              tuple(test, test_gt), data_visualization
              WHERE
              List[Patch] extracted_patches is a list of extracted patches
              List[Patch] extracted_patches_gt is a list with ground truths
                          corresponding to extracted patches
              np.ndarray test is the remaining part of the image, where
                         extracted patches are marked as 0
              np.ndarray test_gt is the ground truth file of the remaining part,
                         which also has extracted patches marked as 0
              np.ndarray data_visualization is a random band presenting
                         positions of the extracted patches
    """
    window = WindowSize(window_size[0], window_size[1])
    input_data = np.load(dataset_path)
    patches = sliding_window(input_data, window)
    gt = np.load(ground_truth_path)
    shuffle(patches)
    total = 0
    extracted_patches = []
    extracted_patches_gt = []
    for patch in patches:
        nonzero = np.count_nonzero(gt[patch.upper_y:patch.lower_y,
                                      patch.left_x:patch.right_x])
        if nonzero == 0:
            continue
        total += nonzero
        extracted_patch = input_data[patch.upper_y:patch.lower_y,
                                     patch.left_x:patch.right_x, :].copy()
        extracted_patch_gt = gt[patch.upper_y:patch.lower_y,
                                patch.left_x:patch.right_x].copy()
        extracted_patches.append(extracted_patch)
        extracted_patches_gt.append(extracted_patch_gt)

        input_data[patch.upper_y:patch.lower_y,
                   patch.left_x:patch.right_x, :] = 0
        gt[patch.upper_y:patch.lower_y,
           patch.left_x:patch.right_x] = 0

        if total >= total_samples_count:
            break
    data_visualization = input_data[:, :, randint(0, input_data.shape[-1])]
    test = input_data
    test_gt = gt
    return (extracted_patches, extracted_patches_gt), (test, test_gt), \
            data_visualization
