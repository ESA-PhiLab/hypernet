import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_length, classes_count):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, classes_count),
        )

    def forward(self, input_data):
        return self.model(input_data)

