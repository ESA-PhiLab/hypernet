import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_shape, classes_count):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_shape + classes_count, 512),
            *block(512, 512),
            *block(512, 512),
            nn.Linear(512, input_shape),
            nn.Sigmoid()
        )

    def forward(self, data, labels):
        data = torch.cat([data, labels], dim=1)
        return self.model(data)
