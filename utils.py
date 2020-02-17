import aenum
import inspect


class Dataset(aenum.Constant):
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


@check_types(str, (list, str))
def load_data(data_path: str, *keys: str):
    """
    Function for loading datasets.

    :param data_path: Path to the data file.
    :param keys: Keys for each dataset.
    """
    print(keys)
