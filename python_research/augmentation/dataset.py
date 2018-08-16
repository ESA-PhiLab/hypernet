import numpy as np
from torch.utils.data import Dataset
from keras.utils import to_categorical

BACKGROUND_LABEL = 0


class HyperspectralDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 ground_truth_path: str,
                 transform=None,
                 normalize: bool=True):
        self.x, self.y = self._transform_data(data_path, ground_truth_path)
        self.transform = transform
        self.classes = np.unique(self.y)
        if normalize:
            self.x = self._normalize()
        self.y = self.y - 1

    @staticmethod
    def _transform_data(data_path, ground_truth_path):
        x = np.load(data_path)
        y = np.load(ground_truth_path)
        transformed = []
        labels = []
        for i, row in enumerate(x):
            for j, pixel in enumerate(row):
                if y[i, j] != BACKGROUND_LABEL:
                    sample = x[i, j, :]
                    transformed.append(sample)
                    labels.append(y[i, j])
        return np.array(transformed).astype(np.float64), np.array(labels)

    def _normalize(self):
        return (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        sample_x = self.x[item, ...]
        sample_y = self.y[item, ...]
        if self.transform:
            sample_x = self.transform(sample_x)
        return sample_x, sample_y
