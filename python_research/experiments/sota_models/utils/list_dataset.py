from torch.utils.data.dataset import Dataset


class ListDataset(Dataset):
    """
    Class that represents datasets as lists of numpy arrays.
    """

    def __init__(self, samples: list, labels: list):
        """
        Set samples and labels instance variables.

        :param samples: List of samples.
        :param labels: List of labels.
        """
        self.samples = samples
        self.labels = labels

    def __getitem__(self, item) -> tuple:
        """
        Dataloader uses this method for loading the batched data.

        :param item: Index of sample
        :return: Given sample based on the index.
        """
        return self.samples[item], self.labels[item]

    def __len__(self) -> int:
        """
        Return length of the data set.

        :return: Length of the data set.
        """
        return len(self.samples)
