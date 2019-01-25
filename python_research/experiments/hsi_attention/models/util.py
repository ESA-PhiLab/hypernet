import torch
import torch.nn as nn
import torch.nn.functional as f
from skimage.transform import resize


def build_convolutional_block(input_channels, output_channels):
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=output_channels, kernel_size=5, padding=2),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(output_channels),
        torch.nn.MaxPool1d(2)
    )


def build_classifier_block(input_size, number_of_classes):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, 512),
        torch.nn.Linear(512, 128),
        torch.nn.Linear(128, number_of_classes),
    )


def build_softmax_module(input_channels):
    return torch.nn.Sequential(
        torch.nn.Conv1d(in_channels=input_channels, out_channels=1, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Softmax(dim=2)
    )


def build_classifier_confidence(input_channels):
    return torch.nn.Sequential(
        nn.Linear(input_channels, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Tanh()
    )


class AttentionBlock(torch.nn.Module):
    def __init__(self, input_channels, input_dimension, num_classes):
        super(AttentionBlock, self).__init__()
        self._softmax_block_1 = build_softmax_module(input_channels)
        self._confidence_net = torch.nn.Linear(input_dimension, 1)
        self._attention_net = torch.nn.Linear(input_dimension, num_classes)
        self._confidence_score = 0.0
        self._prediction = None

    def forward(self, z):
        self._prediction = self._softmax_block_1(z)
        cross_product = torch.einsum("ijk,ilk->ijlk", (self._prediction.clone(), z.clone())) \
            .reshape(self._prediction.shape[0], -1, self._prediction.shape[2])
        cross_product = f.avg_pool1d(cross_product.permute(0, 2, 1), cross_product.shape[1])
        cross_product = cross_product.view(cross_product.shape[0], -1)
        self._confidence_score = f.tanh(self._confidence_net(cross_product))
        return self._attention_net(cross_product) * self._confidence_score

    def get_heatmaps(self, input_size):
        predictions = self._prediction.cpu().data.numpy()
        return [resize(prediction, (1, input_size)) for prediction in predictions]
