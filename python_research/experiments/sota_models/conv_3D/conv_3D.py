import numpy as np
import torch

from python_research.experiments.sota_models.utils.base_module import BaseModule
from python_research.experiments.sota_models.utils.conv3D_utils import conv_block_3d, dense_block, calc_dims


class ConvNet3D(BaseModule):
    def __init__(self, channels: list, input_dim: np.ndarray, dtype: str, batch_size: int, classes: int):
        """
        Convolutional 3D network for hyperspectral data segmentation.

        :param channels: List of channels for output 3D activations.
        :param dtype: Data type.
        :param input_dim: Dimensionality of a single sample in the data set.
        :param batch_size: Size of the batch.
        :param classes: Number of classes.
        """
        super(ConvNet3D, self).__init__(classes=classes)

        assert classes != 0 and batch_size != 0, 'Incorrect number of classes or batch size.' \
                                                 'Please check again both.'

        self.dtype = self.__class__.check_dtype(dtype=dtype)
        self.batch_size = batch_size
        self._block1 = conv_block_3d(channels=channels, dtype=self.dtype[0])
        self._block2 = dense_block(
            num_nodes=calc_dims(input_dim=input_dim, channels=channels[-1]), classes=classes, dtype=self.dtype[0])
        self._cost_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.parameters())

    def forward(self, x, y, val=False, test=False):
        x = torch.unsqueeze(x, dim=1)
        x = self._block1(x)
        x = x.view(self.batch_size, -1)
        x = self._block2(x)
        loss = self._cost_function(x, y)
        arg_max = torch.argmax(x, dim=1)
        accuracy = np.mean((arg_max == y).cpu().numpy())
        if val:
            self.val_accuracies.append(accuracy), self.val_losses.append(loss.clone().detach().cpu().numpy())
        if test:
            self.test_accuracies.append(accuracy), self.test_losses.append(loss.clone().detach().cpu().numpy())
            if self.batch_size != 1:
                for output, label in zip(torch.squeeze(arg_max).cpu().numpy().tolist(),
                                         torch.squeeze(y).cpu().numpy().tolist()):
                    self.acc_per_class[label].append(output == label)
            else:
                self.acc_per_class[int(y.item())].append(arg_max.item() == y.item())
        else:
            self.train_accuracies.append(accuracy), self.train_losses.append(loss.clone().detach().cpu().numpy())
        return loss
