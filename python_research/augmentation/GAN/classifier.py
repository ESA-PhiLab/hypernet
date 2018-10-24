import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Classifier(nn.Module):
    def __init__(self,
                 criterion,
                 input_length: int,
                 classes_count: int,
                 use_cuda: bool=False,
                 patience: int=None,
                 verbose: bool=True):
        """

        :param criterion: PyTorch criterion calculating difference between
                          target and generated values
        :param input_length: Length of the input vector (number of neurons in
                             first layer)
        :param classes_count: Number of classes in the dataset
        :param use_cuda: Whether training should be performed
                         on GPU (True) or CPU (False)
        :param patience: Number of epochs without improvement on accuracy
        :param verbose: Whether to print logs to output
        """
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
        """
        Evaluates whether the training should be terminated based on the
        loss of the discriminator
        :return: Boolean whether the training should be terminated
        """
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
        """
        Performs one training iteration (epoch)
        :param dataloader: Iterable returning a batch of (samples, labels)
                           for each call
        :param optimizer: PyTorch optimization algorithm (from torch.optim)
        :return: None
        """
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

    def _print_metrics(self, epoch: int):
        """
        Prints accuracy to output
        :param epoch: Current epoch
        :return: None
        """
        loss = np.average(self.losses)
        self.losses.clear()
        print("[Epoch {}] [Loss: {}]".format(epoch, loss))

    def train_(self, dataloader: DataLoader, optimizer, epochs: int):
        """
        Performs training of the model until for given number of epochs, or
        stops early based on provided patience
        :param dataloader: Iterable returning a batch of (samples, labels)
                           for each call
        :param optimizer: PyTorch optimization algorithm (from torch.optim)
        :param epochs: Number of epochs the model should be trained
        :return: None
        """
        for epoch in range(epochs):
            self._train_epoch(dataloader, optimizer)
            if self.patience is not None:
                if self._early_stopping():
                    break
            if self.verbose:
                self._print_metrics(epoch)
