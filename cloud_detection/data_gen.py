"""
Generator based data loader for Cloud38 and L8CCA clouds segmentation datasets.
"""

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from pathlib import Path
from tensorflow import keras
from typing import Dict, List, Tuple

from cloud_detection.utils import (
    pad, open_as_array, load_38cloud_gt, load_l8cca_gt,
    strip_nir, load_image_paths
)


class DG_38Cloud(keras.utils.Sequence):
    """
    Data generator for Cloud38 clouds segmentation dataset.
    Works with Keras generators.
    """

    def __init__(
        self,
        files: List[Dict[str, Path]],
        batch_size: int,
        balance_classes: bool = False,
        balance_snow: bool = False,
        dim: Tuple[int, int] = (384, 384),
        shuffle: bool = True,
        with_gt: bool = True,
    ):
        """
        Prepare generator and init paths to files containing image channels.
        :param files: List of dicts containing paths to rgb channels of each
            image in dataset.
        :param batch_size: size of generated batches, only one batch is loaded
            to memory at a time.
        :param balance_classes: if True balance classes.
        :param balance_snow: if True balances patches with snow and clouds.
        :param dim: Tuple with x, y image patches dimensions.
        :param shuffle: if True shuffles dataset before training
                        and on each epoch end.
        :param with_gt: if True returns y along with x.
        """
        self._batch_size: int = batch_size
        self._dim: Tuple[int, int] = dim
        self._shuffle: bool = shuffle
        self._with_gt: bool = with_gt
        self._balance_snow: bool = balance_snow
        self._balance_classes: bool = balance_classes
        self.n_bands: int = len(files[0]) - 1  # -1, because one channel is GT

        self._files: List[Dict[str, Path]] = files
        self._file_indexes = np.arange(len(self._files))
        if self._balance_classes:
            self._balance_file_indexes()
        if self._balance_snow:
            self._balance_snow_indexes()
        if self._shuffle:
            np.random.shuffle(self._file_indexes)

    def _perform_balancing(self, labels: List):
        """
        Perform balancing on given images.
        :param labels: List of pseudo-labels for indexing. For each patch in
                       the dataset should contain 1 if image is to be
                       balanced, otherwise should be 0.
        """
        pos_idx = self._file_indexes[np.array(labels, dtype=bool)]
        neg_idx = self._file_indexes[~np.array(labels, dtype=bool)]
        if len(pos_idx) < len(neg_idx):
            resampled_idx = np.random.choice(pos_idx, len(neg_idx))
            self._file_indexes = np.concatenate(
                [neg_idx, resampled_idx], axis=0)
        elif len(pos_idx) > len(neg_idx):
            resampled_idx = np.random.choice(neg_idx, len(pos_idx))
            self._file_indexes = np.concatenate(
                [pos_idx, resampled_idx], axis=0)
        self._file_indexes = np.sort(self._file_indexes)

    def _balance_file_indexes(self):
        """ Upsamples the file indexes of the smaller class. """
        labels = self._get_labels_for_balancing()
        self._perform_balancing(labels)

    def _balance_snow_indexes(self):
        """ Upsamples the file indexes with snow and clouds. """
        labels = self._get_labels_for_snow_balancing()
        self._perform_balancing(labels)

    def _get_labels_for_snow_balancing(
            self,
            brightness_thr: float = 0.4,
            frequency_thr: float = 0.1
    ) -> List[int]:
        """
        Returns the pseudo-labels for each patch. Pseudo-label being
        1 if certain percent of pixels in patch are above brightness threshold,
        and 0 otherwise.
        :param brightness_thr: brightness threshold of the pixel
                               (in relation to brightest pixel in patch)
                               to classify it as snow.
        :param frequency_thr: frequency threshold of snow pixels to
                              classify patch as snowy.
        :return: list of labels (0 - not snowy, 1 - snowy).
        """
        labels = []
        print(len(self._files))
        for file_ in self._files:
            img = open_as_array(file_)
            if (np.count_nonzero(img > brightness_thr) / img.size) > \
                    frequency_thr:
                labels.append(1)
            else:
                labels.append(0)

        print("Multiplied:", np.count_nonzero(labels))
        return labels

    def _get_labels_for_balancing(self) -> List[int]:
        """
        Returns the pseudo-labels for each patch. Pseudo-label being
        1 if clouds proportion between 0.1 and 0.9, and 0 otherwise.
        """
        labels = []
        for file_ in self._files:
            gt = load_38cloud_gt(file_)
            clouds_prop = np.count_nonzero(gt) / np.prod(self._dim)
            if clouds_prop > 0.1 and clouds_prop < 0.9:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def _data_generation(self, file_indexes_to_gen: np.arange) -> Tuple:
        """
        Generates data containing batch_size samples.
        :param file_indexes_to_gen: Sequence of indexes of files from which
                                    images should be loaded.
        :return: (x, y) (or (x, None) if with_gt is False) data for one batch,
                where x is set of RGB + nir images and y is set of
                 corresponding cloud masks.
        """
        x = np.empty((len(file_indexes_to_gen), *self._dim, 4))
        if self._with_gt:
            y = np.empty((len(file_indexes_to_gen), *self._dim, 1))
        else:
            y = None

        for i, file_index in enumerate(file_indexes_to_gen):
            x[i] = open_as_array(self._files[file_index])
            if self._with_gt:
                y[i] = load_38cloud_gt(self._files[file_index])

        return x, y

    def on_epoch_end(self):
        """
        Triggered after each epoch,
        if shuffle is True randomises file indexing.
        """
        if self._shuffle:
            np.random.shuffle(self._file_indexes)

    def __len__(self) -> int:
        """
        Denotes the number of batches per epoch.
        :return: number of batches per epoch.
        """
        return int(np.ceil(len(self._file_indexes) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray]:
        """
        Generates one batch of data.
        :param index: index of the batch to return.
        :return: (x, y) (or (x, None) if with_gt is False) data for one batch,
                 where x is set of RGB + nir images and y is set of
                 corresponding cloud masks.
        """
        # Generate indexes of the batch
        indexes_in_batch = self._file_indexes[
            index * self._batch_size: (index + 1) * self._batch_size
        ]

        # Generate data
        return self._data_generation(indexes_in_batch)


class DG_L8CCA(keras.utils.Sequence):
    """
    Data generator for L8CCA clouds segmentation dataset.
    Works with Keras generators.
    """

    def __init__(
        self,
        img_paths: Path,
        batch_size: int,
        data_part: Tuple[float] = (0., 1.),
        with_gt: bool = False,
        patch_size: int = 384,
        bands: Tuple[int] = (4, 3, 2, 5),
        bands_names: Tuple[str] = ("red", "green", "blue", "nir"),
        resize: bool = False,
        normalize: bool = True,
        standardize: bool = False,
        shuffle: bool = True,
    ):
        """
        Prepare generator and init paths to files containing image channels.
        :param img_paths: paths to the dirs containing L8CCA images files.
        :param batch_size: size of generated batches, only one batch is loaded
            to memory at a time.
        :param data_part: part of data to include, e.g., (0., 0.2) generates
                          dataloader with samples up to 20-th percentile without
                          20-th percentile, while (0.3, 0.8) generates dataloader
                          with samples from 30-th percentile (including it) up to
                          80-th percentile (without it).
                          (x, 1.) is an exception to the rule, generating dataloader
                          with samples from the x-th percentile (including it) up to
                          AND INCLUDING the last datapoint.
                          To include all samples, use (0., 1.).
                          If shuffle=True, partitions dataset based on shuffled data
                          (with a set seed), else partitions unshuffled dataset.
        :param with_gt: whether to include groundtruth.
        :param patch_size: size of the patches.
        :param bands: band numbers to load
        :param bands_names: names of the bands to load. Should have the same number
                            of elements as bands.
        :param resize: whether to resize img to gt. If True and with_gt=False,
                       will load GT to infer its shape and then delete it.
        :param normalize: whether to normalize the image.
        :param standardize: whether to standardize the image.
        :param shuffle: if True shuffles dataset before training and on each epoch end,
                        else returns dataloader sorted according to img_paths order.
        """
        self._img_paths: Path = img_paths
        self._batch_size: int = batch_size
        self._data_part: Tuple[float] = data_part
        self._with_gt: bool = with_gt
        self._patch_size: int = patch_size
        self._normalize: bool = normalize
        self._standardize: bool = standardize
        self._resize: bool = resize
        self._shuffle: bool = shuffle
        self.n_bands: int = len(bands)
        if self._with_gt or self._resize:
            self.generate_gt_patches()
        self.generate_img_patches(bands=bands, bands_names=bands_names)
        self._patches_indexes = np.arange(len(self.patches))
        if self._data_part != (0., 1.):
            self.partition_data()
        if self._shuffle:
            np.random.shuffle(self._patches_indexes)

    def generate_img_patches(self, bands: Tuple[int] = (4, 3, 2, 5),
                             bands_names: Tuple[str] = (
                                 "red", "green", "blue", "nir"
                                 )):
        """
        Create image patches from the provided bands.
        :param bands: band numbers to load
        :param bands_names: names of the bands to load. Should have the same
                            number of elements as bands.
        """
        self.patches = np.empty(
            (0, self._patch_size, self._patch_size, self.n_bands))
        self.img_shapes = []
        for i, img_path in enumerate(self._img_paths):
            channel_files = {}
            for name, band in zip(bands_names, bands):
                channel_files[name] = list(img_path.glob(f"*_B{band}.TIF"))[0]
            img = open_as_array(channel_files=channel_files,
                                channel_names=bands_names,
                                size=self.original_gt_shapes[i]
                                if self._resize else None,
                                normalize=self._normalize,
                                standardize=self._standardize)
            img = pad(img, self._patch_size)
            self.img_shapes.append(img.shape)
            img_patches = rearrange(
                img, "(r dr) (c dc) b -> (r c) dr dc b",
                dr=self._patch_size, dc=self._patch_size
            )
            self.patches = np.concatenate((self.patches, img_patches))
        del img

    def generate_gt_patches(self):
        """
        Create GT patches.
        """
        if self._with_gt:
            self.gt_patches = np.empty(
                (0, self._patch_size, self._patch_size, 1)
                )
        self.original_gt_shapes = []
        for img_path in self._img_paths:
            gt = load_l8cca_gt(path=img_path)
            self.original_gt_shapes.append(gt.shape)
            if self._with_gt:
                gt = pad(gt, self._patch_size)
                img_gt_patches = rearrange(
                    gt, "(r dr) (c dc) 1 -> (r c) dr dc 1",
                    dr=self._patch_size, dc=self._patch_size
                )
                self.gt_patches = np.concatenate(
                    (self.gt_patches, img_gt_patches)
                    )
        del gt

    def partition_data(self, seed=42):
        """
        Partition data based on data_part arg.
        :param seed: random seed.
        """
        if self._shuffle:
            saved_seed = np.random.get_state()
            np.random.seed(seed)
            np.random.shuffle(self._patches_indexes)
            np.random.set_state(saved_seed)
        assert len(self._data_part) == 2
        for perc in self._data_part:
            assert (type(perc) is float) and (perc >= 0.) and (perc <= 1.)
        from_, to_ = self._data_part
        idx_from = int(from_ * len(self._patches_indexes))
        idx_to = int(to_ * len(self._patches_indexes))
        if to_ < 1.:
            self._patches_indexes = self._patches_indexes[idx_from:idx_to]
        elif to_ == 1.:
            self._patches_indexes = self._patches_indexes[idx_from:]
        self.patches = self.patches[self._patches_indexes]
        if self._with_gt:
            self.gt_patches = self.gt_patches[self._patches_indexes]
        self._patches_indexes = np.arange(len(self.patches))

    def on_epoch_end(self):
        """
        Triggered after each epoch,
        if shuffle is True randomises file indexing.
        """
        if self._shuffle:
            np.random.shuffle(self._patches_indexes)

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        :return: number of batches per epoch.
        """
        return int(np.ceil(len(self._patches_indexes) / self._batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, None]:
        """
        Generates one batch of data.
        :param index: index of the batch to return.
        :return: (x, None) data for one batch,
                 where x is set of RGB + nir images.
        """
        indexes_in_batch = self._patches_indexes[
            index * self._batch_size: (index + 1) * self._batch_size
        ]
        if self._with_gt:
            y = self.gt_patches[indexes_in_batch]
        else:
            y = None
        return (
            self.patches[indexes_in_batch],
            y,
        )


def main_38Cloud():
    """ Demo 38Cloud data loading. """
    base_path = Path("datasets/clouds/38-Cloud/38-Cloud_training")

    split_names = ("train", "validation", "test")
    splits = load_image_paths(
        base_path=base_path, split_ratios=(0.8, 0.15, 0.05))

    for name, split in zip(split_names, splits):
        dg = DG_38Cloud(files=split, batch_size=16)
        sample_batch_x, sample_batch_y = dg[3]
        sample_batch_y = sample_batch_y[:, :, :, 0]

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(strip_nir(sample_batch_x[0]))
        plt.title(f"Split: { name }\n sample image")
        plt.subplot(1, 3, 2)
        plt.imshow(sample_batch_y[0])
        plt.title(f"Split: { name }\n sample gt mask")
    plt.show()


def main_L8CCA():
    """ Demo L8CCA data loading. """
    base_path = Path(
        "datasets/clouds/"
        + "Landsat-Cloud-Cover-Assessment-Validation-Data-Partial"
        )
    img_paths = [base_path / "Barren" / "LC81390292014135LGN00",
                 base_path / "Forest" / "LC80160502014041LGN00"]

    split_names = ("train", "validation", "test")
    splits = ((0., 0.8), (0.8, 0.95), (0.95, 1.))

    for name, split in zip(split_names, splits):
        dg = DG_L8CCA(img_paths=img_paths, batch_size=16,
                      data_part=split, with_gt=True)
        sample_batch_x, sample_batch_y = dg[2]
        sample_batch_y = sample_batch_y[:, :, :, 0]

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(strip_nir(sample_batch_x[0]))
        plt.title(f"Split: { name }\n sample image")
        plt.subplot(1, 3, 2)
        plt.imshow(sample_batch_y[0])
        plt.title(f"Split: { name }\n sample gt mask")
    plt.show()


if __name__ == "__main__":
    print("38Cloud demo")
    main_38Cloud()
    print("L8CCA demo")
    main_L8CCA()
