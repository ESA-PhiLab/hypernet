import torch


def build_block1(in_channels: int, out_channels: int, dtype: torch.Tensor) -> torch.nn.Sequential:
    """
    Applies a spatial convolution with receptive filed of (1 x 1) over an input signal.
    Each filter extends throughout the entire spectral axis of the input volume.

    :param in_channels: Number of input channels.
    :param out_channels: Number of feature maps.
    :param dtype: Data type used by the model.
    :return: Sequential container which stores all modules.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
        torch.nn.ReLU()
    ).type(dtype)


def build_block2(dtype: torch.Tensor, neighborhood_size: int) -> torch.nn.Sequential:
    """
    Architectural design of the second block of configuration 4.
    Applies a spectral-wise 1D convolution over an input signal.

    :param dtype: Data type used by the model.
    :param neighborhood_size: Spatial size of the sample.
    :return: Sequential container which stores all modules.
    """
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=neighborhood_size ** 2, out_channels=20, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3),
        torch.nn.ReLU(),
        torch.nn.Conv1d(in_channels=10, out_channels=5, kernel_size=5),
        torch.nn.ReLU()
    ).type(dtype)


def build_block3(entries: int, num_of_classes: int, dtype: torch.Tensor) -> torch.nn.Sequential:
    """
    Return classifier sequential block to produce logits for each class respectively.

    :param entries: Number of entries into the dense layer.
    :param num_of_classes: Number of output neural activations.
    :param dtype: Data type used by the model.
    :return: Sequential container which stores all modules.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(entries, 100),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(100, num_of_classes),
    ).type(dtype)
