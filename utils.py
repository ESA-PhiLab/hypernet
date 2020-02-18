import inspect
import typing

import aenum
import h5py
import numpy as np


class Dataset(aenum.Constant):
    SAMPLES_DIM = 0
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    DATA = 'data'
    LABELS = 'labels'


class Model(aenum.Constant):
    TRAINED_MODEL = 'trained_model'


def check_types(*types):
    def function_wrapper(function):
        assert len(types) == len(inspect.signature(function).parameters), \
            'Number of arguments must match the number of possible types.'

        def validate_types(*args, **kwargs):
            for (obj, type_) in zip(args, types):
                assert isinstance(obj, type_), \
                    'Object {0} does not match {1} type.'.format(obj, type_)
            # If all objects are consistent return function:
            return function(*args, **kwargs)
        return validate_types
    return function_wrapper


@check_types(str, str)
def load_data(data_path, *keys: str) -> typing.List[typing.Dict]:
    """
    Function for loading datasets as list of dictionaries.

    :param data_path: Path to the dataset.
    :param keys: Keys for each dataset.
    """
    raw_data = h5py.File(data_path, 'r')
    datasets = []
    for dataset_key in keys:
        datasets.append({
            Dataset.DATA: np.asarray(raw_data[dataset_key][Dataset.DATA]),
            Dataset.LABELS: np.asarray(
                raw_data[dataset_key][Dataset.LABELS])
        })
    return datasets
