import aenum


class Dataset(aenum.Constant):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'
    DATA = 'data'
    LABELS = 'labels'


class SatelliteH5Keys(aenum.Constant):
    CHANNELS = 'channels'
    CUBE = 'mean'
    COV = 'cov'
    GT_TRANSFORM_MAT = 'to_earth_transform'


class DataStats(aenum.Constant):
    MIN = 'min'
    MAX = 'max'
