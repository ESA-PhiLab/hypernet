from torch.utils.data.dataset import Dataset


class ListDataset(Dataset):
    """
    Class that represents data sets as lists of numpy arrays.
    """

    def __init__(self, samples: list, labels: list):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, item):
        """
        Dataloader uses this method for loading the batched data.
        :param item: Index of sample
        :return: Given sample based on the index.
        """
        return self.samples[item], self.labels[item]

    def __len__(self):
        return len(self.samples)
