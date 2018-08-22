import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_shape, 128),
            *block(128, 256),
            *block(256, 512),
            *block(512,1024),
            nn.Linear(1024, input_shape - 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)
