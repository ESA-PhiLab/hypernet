from typing import Tuple, NamedTuple


class TrainTestIndices(NamedTuple):
    train_indices: dict
    test_indices: dict


class ModelSettings(NamedTuple):
    input_neighbourhood: Tuple[int, int]
    first_conv_kernel_size: Tuple[int, int]
    max_pooling_strides: Tuple[int, int]
    last_layer_conv_padding: str
