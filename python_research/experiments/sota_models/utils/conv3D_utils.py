import numpy as np
import torch


def dense_block(num_nodes: int, classes: int, dtype: torch.Tensor) -> torch.nn.Sequential:
    """
    Build fully-connected block.

    :param num_nodes: Number of total input nodes.
    :param classes: Number of classes.
    :param dtype: Data type used by the model.
    :return: Sequential container which stores all modules.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(num_nodes, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features=128, out_features=classes)
    ).type(dtype)


def conv_block_3d(channels: list, dtype: torch.Tensor) -> torch.nn.Sequential:
    """
    Three-layer convolutional block.
    The kernel dimensionality in each layer is: (3 x 3 x 3)
    Stride is: (1 x 1 x 1), i.e. "unit stride".
    Zero padding.

    :param channels: List of channels in each layer.
    :param dtype: Data type used by the model.
    :return: Sequential container which stores all modules.
    """
    return torch.nn.Sequential(
        torch.nn.Conv3d(in_channels=1, out_channels=channels[0], kernel_size=(3, 3, 3), stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(in_channels=channels[0], out_channels=channels[1], kernel_size=(3, 3, 3), stride=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(in_channels=channels[1], out_channels=channels[2], kernel_size=(3, 3, 3), stride=1),
        torch.nn.ReLU(),
    ).type(dtype)


def calculate_dim(input_size: np.ndarray, padding: int, kernel_size: int, stride: int) -> np.ndarray:
    """
    Calculate output dimensionality of tensor.

    :param input_size: Input size of the sample.
    :param padding: Size of the padding.
    :param kernel_size: Size of the kernel.
    :param stride: Stride for kernel.
    :return: Output size of the sample.
    """
    return (((input_size + 2 * padding - (kernel_size - 1) - 1) / stride) + 1).astype(int)


def calc_dims(input_dim: list, channels: int) -> int:
    """
    Calculate number of nodes for the dense layer.

    :param input_dim: Dimensionality of an input sample.
    :param channels: Number of channels in the last layer.
    :return: Number of total input nodes to set in the linear layer.
    """
    num_nodes = calculate_dim(input_size=input_dim, kernel_size=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padding=np.array([0, 0, 0]))
    num_nodes = calculate_dim(input_size=num_nodes, kernel_size=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padding=np.array([0, 0, 0]))
    num_nodes = calculate_dim(input_size=num_nodes, kernel_size=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padding=np.array([0, 0, 0]))
    return int(np.floor(num_nodes).prod() * channels)
