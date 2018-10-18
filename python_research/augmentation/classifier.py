import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self,
                 criterion,
                 input_length,
                 classes_count,
                 use_cuda: bool=False,
                 patience: int=None,
                 verbose: bool=True):
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
        self.criterion = criterion
        self.use_cuda = use_cuda
        self.patience = patience
        self.verbose = verbose
        self.best_loss = np.inf
        self.epochs_without_improvement = 0
        self.losses = []

    def forward(self, input_data):
        return self.model(input_data)

    def _early_stopping(self) -> bool:
        if np.average(self.losses) < self.best_loss:
            self.best_loss = np.average(self.losses)
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                if self.verbose:
                    print("{} epochs without improvement, "
                          "terminating".format(self.patience))
                return True
            return False

    def _train_epoch(self, dataloader, optimizer):
        for samples, labels in dataloader:
            samples = Variable(samples).type(torch.FloatTensor)
            labels = Variable(labels).type(torch.LongTensor)
            if self.use_cuda:
                samples = samples.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            validity = self.forward(samples)
            loss = self.criterion(validity, labels)
            loss.backward()
            optimizer.step()
            if self.verbose:
                self.losses.append(loss.item())

    def _print_metrics(self, epoch):
        loss = np.average(self.losses)
        self.losses.clear()
        print("[Epoch {}] [Loss: {}]".format(epoch, loss))

    def train_(self, dataloader: DataLoader, optimizer, epochs: int):
        for epoch in range(epochs):
            self._train_epoch(dataloader, optimizer)
            if self.patience is not None:
                if self._early_stopping():
                    break
            if self.verbose:
                self._print_metrics(epoch)
