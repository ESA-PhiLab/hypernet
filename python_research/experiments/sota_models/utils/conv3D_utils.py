import numpy as np
import torch


def dense_block(num_nodes, classes, dtype):
    """
    Build fully - connected block.

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


def conv_block_3d(channels: list, dtype):
    """
    Three - layer convolutional block.
    The kernel dimensionality in each layer is: (3 x 3 x 3)
    Stride is: (1 x 1 x 1)

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


def calculate_dim(in_x, padd, kernel, stride):
    out_x = ((in_x + 2 * padd - (kernel - 1) - 1) / stride) + 1
    return out_x


def calc_dims(input_dim, channels):
    """
    Calculate dimensions for the dense layer.

    :param input_dim: Dimensionality of an input sample.
    :param channels: Size of the last channel.
    :return: Number of total input nodes to set in the linear layer.
    """
    num_nodes = calculate_dim(in_x=input_dim, kernel=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padd=np.array([0, 0, 0]))
    num_nodes = calculate_dim(in_x=num_nodes, kernel=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padd=np.array([0, 0, 0]))
    num_nodes = calculate_dim(in_x=num_nodes, kernel=np.array([3, 3, 3]),
                              stride=np.array([1, 1, 1]), padd=np.array([0, 0, 0]))
    return int(np.floor(num_nodes).prod() * channels)
