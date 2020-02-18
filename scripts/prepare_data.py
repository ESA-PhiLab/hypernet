"""
Load the data, reformat it to have right dimensions, split it into train, test 
and val sets and save them in .h5 file with 'train', 'val' and 'test' goups,
each having 'data' and 'labels' keys.
"""

import os
import h5py

import clize

from input_output import load_npy
from utils import Dataset, train_val_test_split, reshape_to_1d_samples

EXTENSION = 1


def main(*,
         data_file_path: str,
         ground_truth_path: str,
         output_path: str,
         train_size: float = 0.8,
         val_size: float = 0.1,
         balanced: bool = True,
         background_label: int = 0,
         channels_idx: int = 0):
    """
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the data file.
    :param output_path: Path under in which the output .h5 file will be stored
    :type output_dir: Directory in which the prepared .h5 file will be stored
    :param train_size: If float, should be between 0.0 and 1.0, if balanced = True, it represents percentage of each class to be extracted, 
                       If float and balanced = False, it represents percetange of the whole dataset to be extracted with samples drawn randomly. 
                       If int and balanced = True, it represents number of samples to be drawn from each class. 
                       If int and balanced = False, it represents overall number of samples to be drawn, randomly. 
                       Defaults to 0.8
    :type train_size: float or int
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of each class from the training set 
                     to be extracted as a validation set, defaults to 0.1
    :param balanced: Indicated whether the extracted training set should be balanced, defaults to True
    :param background_label: Label indicating the background in GT file
    :param channels_idx: Index specifing the dimensions of channels
    :raises TypeError: When provided data or labels file is not supported
    """
    if data_file_path.endswith('.npy') and ground_truth_path.endswith('.npy'):
        data, labels = load_npy(data_file_path, ground_truth_path)
    else:
        raise TypeError("The following data file type is not supported: {}".format(
            os.path.splitext(data_file_path)[EXTENSION]))

    data, labels = reshape_to_1d_samples(data, labels, channels_idx)
    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(
        data, labels, train_size, val_size, balanced, background_label)
    
    data_file = h5py.File(output_path, 'w')
    train_group = data_file.create_group(Dataset.TRAIN)
    train_group.create_dataset(Dataset.DATA, data=train_x)
    train_group.create_dataset(Dataset.LABELS, data=train_y)

    val_group = data_file.create_group(Dataset.VAL)
    val_group.create_dataset(Dataset.DATA, data=val_x)
    val_group.create_dataset(Dataset.LABELS, data=val_y)
    
    test_group = data_file.create_group(Dataset.TEST)
    test_group.create_dataset(Dataset.DATA, data=test_x)
    test_group.create_dataset(Dataset.LABELS, data=test_y)
    data_file.close()


if __name__ == '__main__':
    clize.run(main)
