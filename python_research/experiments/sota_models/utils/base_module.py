import numpy as np
import torch


class BaseModule(torch.nn.Module):
    """
    Base class for models, which implements several helper methods concerning artifacts.
    """

    def __init__(self, classes):
        """
        Declare instance variables designed to hold results.

        :param classes: Number of classes.
        """
        super(BaseModule, self).__init__()

        self.train_accuracies = []
        self.train_losses = []
        self.train_acc_history = []
        self.train_loss_history = []

        self.val_accuracies = []
        self.val_losses = []
        self.val_acc_history = []
        self.val_loss_history = []

        self.test_accuracies = []
        self.test_losses = []
        self.test_acc_history = []
        self.test_loss_history = []

        self.acc_per_class = [[] for _ in range(classes)]

    def forward(self, *x) -> torch.Tensor:
        pass

    @staticmethod
    def check_dtype(dtype: str) -> list:
        """
        Check and evaluate tpe of data on which the model will operate.

        :param dtype: Data type.
        :return: Adjusted data types in list.
        """
        if dtype == "torch.cuda.FloatTensor":
            return [torch.cuda.FloatTensor, torch.cuda.LongTensor]
        elif dtype == "torch.FloatTensor":
            return [torch.FloatTensor, torch.LongTensor]

    def get_train_results(self) -> tuple:
        """
        Get training results.

        :return: Training results.
        """
        accuracy = np.mean(np.asarray(self.train_accuracies))
        self.train_acc_history.append(accuracy)
        loss = np.mean(np.asarray(self.train_losses))
        self.train_loss_history.append(loss)
        self.train_accuracies = []
        self.train_losses = []
        return accuracy, loss

    def get_val_results(self) -> tuple:
        """
        Get validation results.

        :return: Validation results.
        """
        accuracy = np.average(np.asarray(self.val_accuracies))
        self.val_acc_history.append(accuracy)
        loss = np.mean(np.asarray(self.val_losses))
        self.val_loss_history.append(loss)
        self.val_accuracies = []
        self.val_losses = []
        return accuracy, loss

    def get_test_results(self) -> tuple:
        """
        Get testing results.

        :return: Testing results.
        """
        accuracy = np.mean(np.asarray(self.test_accuracies))
        self.test_acc_history.append(accuracy)
        loss = np.mean(np.asarray(self.test_losses))
        self.test_loss_history.append(loss)
        self.test_accuracies = []
        self.test_losses = []
        return accuracy, loss
