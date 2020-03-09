"""
Load the data, reformat it to have [SAMPLES, ....] dimensions,
split it into train, test and val sets and save them in .h5 file with
'train', 'val' and 'test' groups, each having 'data' and 'labels' keys.
"""

import os
import h5py
import numpy as np

import clize

import ml_intuition.data.preprocessing as preprocessing
import ml_intuition.data.io as io
import ml_intuition.data.utils as utils

EXTENSION = 1


def main(*,
         data_file_path: str,
         ground_truth_path: str,
         output_path: str,
         train_size: float = 0.8,
         val_size: float = 0.1,
         stratified: bool = True,
         background_label: int = 0,
         channels_idx: int = 0):
    """
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the data file.
    :param output_path: Path under in which the output .h5 file will be stored
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage of each
                        class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :type train_size: float or int
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param background_label: Label indicating the background in GT file
    :param channels_idx: Index specifying the channels position in the provided
                         data
    :raises TypeError: When provided data or labels file is not supported
    """
    if data_file_path.endswith('.npy') and ground_truth_path.endswith('.npy'):
        data, labels = io.load_npy(data_file_path, ground_truth_path)
        data, labels = preprocessing.reshape_cube_to_2d_samples(
            data, labels, channels_idx)
    elif data_file_path.endswith('.h5') and ground_truth_path.endswith('.tiff'):
        data, gt_transform_mat = io.load_satellite_h5(data_file_path)
        labels = io.load_tiff(ground_truth_path)
        data_2d_shape = data.shape[1:]
        labels = preprocessing.align_ground_truth(data_2d_shape, labels,
                                                  gt_transform_mat)
        data, labels = preprocessing.reshape_cube_to_2d_samples(data, labels,
                                                                channels_idx)
        data, labels = preprocessing.remove_nan_samples(data, labels)
    else:
        raise TypeError(
            "The following data file type is not supported: {}".format(
                os.path.splitext(data_file_path)[EXTENSION]))

    data = data[labels != background_label]
    labels = labels[labels != background_label]
    labels = preprocessing.normalize_labels(labels)

    train_x, train_y, val_x, val_y, test_x, test_y = utils.train_val_test_split(
        data, labels, train_size, val_size, stratified)

    data_file = h5py.File(output_path, 'w')

    train_min, train_max = np.amin(train_x), np.amax(train_x)
    data_file.attrs.create(utils.DataStats.MIN, train_min)
    data_file.attrs.create(utils.DataStats.MAX, train_max)

    train_group = data_file.create_group(utils.Dataset.TRAIN)
    train_group.create_dataset(utils.Dataset.DATA, data=train_x)
    train_group.create_dataset(utils.Dataset.LABELS, data=train_y)

    val_group = data_file.create_group(utils.Dataset.VAL)
    val_group.create_dataset(utils.Dataset.DATA, data=val_x)
    val_group.create_dataset(utils.Dataset.LABELS, data=val_y)

    test_group = data_file.create_group(utils.Dataset.TEST)
    test_group.create_dataset(utils.Dataset.DATA, data=test_x)
    test_group.create_dataset(utils.Dataset.LABELS, data=test_y)
    data_file.close()


if __name__ == '__main__':
    clize.run(main)
