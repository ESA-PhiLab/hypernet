from typing import Tuple, NamedTuple


class TrainTestIndices(NamedTuple):
    train_indices: dict
    val_indices: dict
    test_indices: dict


class ModelSettings(NamedTuple):
    input_neighborhood: Tuple[int, int]
    first_conv_kernel_size: Tuple[int, int]
