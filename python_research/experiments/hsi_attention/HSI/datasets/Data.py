import scipy.io as sio
import numpy as np
import torch
import random as rd
from collections import Counter


class LoadContent(object):
    """
    This class is for loading input features and labels for the network.

    Disclaimer:

    - Works with matlab files.
    - User need to provide a key for getting the content out of the file.
    """

    def __init__(self, path_content: str, path_labels: str,
                 content_key: str, labels_key: str):
        """
        Init method for the class.

        :param path_content: Path to the input features.
        :param path_labels:  Path to the labels.
        :param content_key:  Key for reading the matlab file.
        :param labels_key:  Key for reading the matlab file.
        """
        self.content_path = path_content
        self.labels_path = path_labels

        self.content_key = content_key
        self.labels_key = labels_key

        self.list_of_content = []
        self.list_of_labels = []

    def load_content(self) -> tuple:
        """
        Load the content with labels and returns it.
        Also, converts the lists to tensors.

        :return: List of input features and labels.
        """
        # Loads only those two parts from matlab file.
        mat_content = sio.loadmat(self.content_path)[self.content_key]
        mat_labels = sio.loadmat(self.labels_path)[self.labels_key]

        num_of_classes = np.asarray(mat_labels).max()
        # print(np.asarray(mat_content).max(), np.asarray(mat_content).min())

        for i in range(int(np.asarray(mat_content).shape[0])):

            for j in range(int(np.asarray(mat_content).shape[1])):

                # Appends each pixel with its whole channel dimension.
                self.list_of_content\
                    .append(torch
                            .from_numpy(np.asarray(mat_content[:][i][j])
                            .astype(float)
                            .reshape(1, 1, mat_content.shape[2])))
                # Shape for the input data should be (1, 1, num_of_bands)

                self.list_of_labels.append(LoadContent.
                                           one_hot_encoder(mat_labels[i][j],
                                                            num_of_classes))
        return self.list_of_content, self.list_of_labels

    @staticmethod
    def one_hot_encoder(label: int, c: int) -> torch.Tensor:
        """
        One hot encoding the labels.

        :param label: Number of label.
        :param c: Number of classes.
        :return:
        """

        assert c > 1, 'Number of classes must be greater than one.'

        one_hot_label = torch.zeros(1, c)

        if label != 0:
            one_hot_label[0][label-1] = 1
        else:
            one_hot_label[0][0] = 1

        return one_hot_label

    @staticmethod
    def randomize_content(x: list, y: list) -> tuple:
        """
        Randomize features and labels.

        :param x: List of input features.
        :param y: List of labels.

        :return: Tuple of randomized input features and labels.
        """

        tmp = list(zip(x, y))
        rd.shuffle(tmp)

        x, y = zip(*tmp)

        return x, y

    @staticmethod
    def make_sets(x, y):
        """
        Create sets for training, validation and testing.

        :param x: List of input features.
        :param y: List of labels.

        :return: Tuple of randomized input features and labels divided to sets, (validation... etc.).
        """

        for i in range(len(y)):
            y[i] = y[i].numpy()

        y = np.asarray(y)

        unique, counts = np.unique(y, return_counts=True, axis=0)

        #nums = dict(zip(unique, counts))

        v = np.asarray(counts).sum()

        print(unique, counts)

        #i = [i for i, b in enumerate(y)]



