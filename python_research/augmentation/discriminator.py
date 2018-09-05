import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_shape, classes_count):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape + classes_count, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, input_data, labels):
        input_data = torch.cat([input_data, labels], dim=1)
        return self.model(input_data)
