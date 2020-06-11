import enum

import aenum


class Sample(enum.IntEnum):
    SAMPLES_DIM = 0
    FEATURES_DIM = 1


class Experiment(aenum.Constant):
    INFERENCE_METRICS = 'inference_metrics.csv'
    INFERENCE_GRAPH_METRICS = 'inference_graph_metrics.csv'
    INFERENCE_FAIR_METRICS = 'inference_fair_metrics.csv'
    EXPERIMENT = 'experiment'
    REPORT = 'report.csv'
    REPORT_FAIR = 'report-fair.csv'


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


class NodeNames(aenum.Constant):
    INPUT = 'input_node'
    OUTPUT = 'output_node'


class MLflowTags(aenum.Constant):
    SPLIT = 'split'
    FOLD = 'fold'
    QUANTIZED = 'quantized'


class Splits(aenum.Constant):
    BALANCED = 'balanced'
    IMBALANCED = 'imbalanced'
    GRIDS = 'grids'
