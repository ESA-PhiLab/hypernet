import numpy as np
import torch

from python_research.experiments.sota_models.utils.base_module import BaseModule
from python_research.experiments.sota_models.utils.bass_utils import build_block1, build_block2, build_block3


class Bass(BaseModule):
    def __init__(self, classes: int, nb: int, in_channels_in_block1: int, out_channels_in_block1: int,
                 neighborhood_size: int, batch_size: int, dtype: str, lr=0.0005):
        """
        BASS model. (Configuration 4).
        Cost function: CrossEntropyLoss.
        Optimizer: Adam, (lr=0.0005).

        :param classes: Number of classes.
        :param nb: Number of convolutional blocks.
        :param in_channels_in_block1: Number of input channels for first block of the network.
        :param out_channels_in_block1: Number of output channels for first block of the network.
        :param neighborhood_size: Spatial size of samples.
        :param batch_size: Size of the batch.
        :param dtype: Data type used by the model as string.
        :param lr: Learning rate hyperparameter for the optimizer. (The default is 0.0005.)
        """
        super(Bass, self).__init__(classes=classes)
        assert out_channels_in_block1 % nb == 0, \
            "Number of output channels for the first block must be divisible by the number of convolutional blocks."
        self.dtype = self.__class__.check_dtype(dtype=dtype)
        self._nb = nb
        self._batch_size = batch_size
        self._block2 = torch.nn.ModuleList()
        self._final_band_size = int(((out_channels_in_block1 / nb) - 10) * 5)

        self._block1 = build_block1(in_channels=in_channels_in_block1,
                                    out_channels=out_channels_in_block1,
                                    dtype=self.dtype[0])

        self._block3 = build_block3(entries=nb * self._final_band_size, num_of_classes=classes,
                                    dtype=self.dtype[0])
        [self._block2.append(build_block2(dtype=self.dtype[0],
                                          neighborhood_size=neighborhood_size)) for _ in range(nb)]
        self._cost_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def forward(self, x, y, val=False, test=False) -> torch.Tensor:
        """
        Feed forward method.

        :param x: Input sample.
        :param y: Target label for the sample.
        :param val: Set to "True" during validation phase.
        :param test: Set to "False" during inference phase.
        :return: Loss of the model over given batch.
        """
        x = self._block1(x)
        x = torch.split(x, int(x.shape[1] / self._nb), dim=1)
        x = [x_.view(self._batch_size, x_.shape[1], -1) for x_ in x]
        x = [x_.permute(0, 2, 1) for x_ in x]
        x = torch.cat(tuple([self._block2[i](split) for i, split in enumerate(x)]), 1).view(self._batch_size, -1)
        x = self._block3(x)
        loss = self._cost_function(x, y)
        arg_max = torch.argmax(x, dim=1)
        accuracy = np.mean((arg_max == y).cpu().numpy())
        if val:
            self.val_accuracies.append(accuracy), self.val_losses.append(loss.clone().detach().cpu().numpy())
        if test:
            self.test_accuracies.append(accuracy), self.test_losses.append(loss.clone().detach().cpu().numpy())
            if self._batch_size != 1:
                for output, label in zip(torch.squeeze(arg_max).cpu().numpy().tolist(),
                                         torch.squeeze(y).cpu().numpy().tolist()):
                    self.acc_per_class[label].append(output == label)
            else:
                self.acc_per_class[int(y.item())].append(arg_max.item() == y.item())
        else:
            self.train_accuracies.append(accuracy), self.train_losses.append(loss.clone().detach().cpu().numpy())
        return loss
