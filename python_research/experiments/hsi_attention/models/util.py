import numpy as np
import torch
import torch.nn.functional as f
from skimage.transform import resize


def build_convolutional_block(input_channels: int, output_channels: int) -> torch.nn.Sequential:
    """
    Create convolutional block for the network.

    :param input_channels: Number of input feature maps.
    :param output_channels: Number of designed output feature maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(output_channels),
        torch.nn.MaxPool1d(2)
    )


def build_classifier_block(input_size: int, number_of_classes: int) -> torch.nn.Sequential:
    """
    Build classifier block designed for obtaining the prediction.

    :param input_size: Input number of features.
    :param number_of_classes: Number of classes.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, number_of_classes)
    )


def build_softmax_module(input_channels: int) -> torch.nn.Sequential:
    """
    Build softmax module for attention mechanism.

    :param input_channels: Number of input feature maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=1, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=2)
    )


def build_classifier_confidence(input_channels: int) -> torch.nn.Sequential:
    """
    Create sequential module for classifier confidence score.

    :param input_channels: Number of input activation maps.
    :return: Sequential module.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(input_channels, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.Tanh()
    )


class AttentionBlock(torch.nn.Module):
    def __init__(self, input_channels: int, input_dimension: int, num_classes: int):
        """
        Attention module constructor.

        :param input_channels: Number of input activation maps.
        :param input_dimension: Size of input spectral dimensionality.
        :param num_classes: Number of classes.
        """
        super(AttentionBlock, self).__init__()
        self._softmax_block_1 = build_softmax_module(input_channels)
        self._confidence_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 1),
            torch.nn.Tanh()
        )
        self._attention_net = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, num_classes)
        )
        self._attention_heatmaps = [[] for _ in range(num_classes)]

    def forward(self, z: torch.Tensor, y: torch.Tensor, infer: bool) -> torch.Tensor:
        """
        Feed forward method for attention module.

        :param z: Input tensor.
        :param y: Labels.
        :param infer: Boolean variable indicating whether to save attention heatmap which is later used in the
                      band selection process.
        :return: Weighted attention module hypothesis.
        """
        heatmap = self._softmax_block_1(z)
        if infer:
            for i, class_ in enumerate(y):
                self._attention_heatmaps[torch.argmax(class_)].append(heatmap[i])
        cross_product = torch.einsum("ijk,ilk->ijlk", (heatmap.clone(), z.clone())) \
            .reshape(heatmap.shape[0], -1, heatmap.shape[2])
        cross_product = f.avg_pool1d(cross_product.permute(0, 2, 1), cross_product.shape[1])
        cross_product = cross_product.squeeze()
        return self._attention_net(cross_product) * self._confidence_net(cross_product)

    def get_heatmaps(self, input_size: int) -> np.ndarray:
        """
        Return heatmaps given initial spectral size.
        The heatmaps are interpolated to specified spectral size.

        :param input_size: Designed spectral size.
        :return: Array containing heatmaps collected by the attention module.
        """
        for i in range(self._attention_heatmaps.__len__()):
            for j in range(self._attention_heatmaps[i].__len__()):
                self._attention_heatmaps[i][j] = resize(self._attention_heatmaps[i][j].cpu().numpy(), (1, input_size))
        for i in range(self._attention_heatmaps.__len__()):
            self._attention_heatmaps[i] = np.mean(self._attention_heatmaps[i], axis=0)
        return np.asarray(self._attention_heatmaps)
