from enum import Enum


class Dataset(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def check_types(*types):
    def function_wrapper(function):
        assert len(types) == function.__code__.co_argcount, \
            'Number of arguments must match the number of possible types.'

        def validate_types(*args, **kwargs):
            for (obj, type_) in zip(args, types):
                assert isinstance(obj, type_), \
                    'Object {0} does not match {1} type.'.format(obj, type_)
            # If all objects are consistent return function:
            return function(*args, **kwargs)
        return validate_types
    return function_wrapper
