""" Generator based data loader for Cloud38 clouds segmentation dataset. """

import math

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from typing import Dict, List, Tuple

from cloud_detection.utils import pad


def strip_nir(hyper_img: np.ndarray) -> np.ndarray:
    """
    Strips nir channel so image can be displayed.
    :param hyper_img: image with shape (x, y, 4) where fourth channel is nir.
    :return: image with shape (x, y, 3) with standard RGB channels.
    """
    return hyper_img[:, :, :3]


def load_image_paths(
    base_path: Path,
    patches_path: Path = None,
    split_ratios: List[float] = [1.0],
    shuffle: bool = True,
    img_id: str = None,
) -> List[List[Dict[str, Path]]]:
    """
    Build paths to all files containing image channels.
    :param base_path: root path containing directories with image channels.
    :param patches_path: path to images patches names to load
    :param split_ratios: list containing split ratios,
                         splits should add up to one.
    :param shuffle: whether to shuffle image paths.
    :param img_id: image ID; if specified, load paths for this image only.
    :return: list with paths to image files, separated into splits.
        Structured as: list_of_splits[list_of_files['file_channel', Path]]
    """

    def combine_channel_files(red_file: Path) -> Dict[str, Path]:
        """
        Get paths to 'green', 'blue', 'nir' and 'gt' channel files
        based on path to the 'red' channel of the given image.
        :param red_file: path to red channel file.
        :return: dictionary containing paths to files with each image channel.
        """
        return {
            "red": red_file,
            "green": Path(str(red_file).replace("red", "green")),
            "blue": Path(str(red_file).replace("red", "blue")),
            "nir": Path(str(red_file).replace("red", "nir")),
            "gt": Path(str(red_file).replace("red", "gt")),
        }

    def build_paths(
        base_path: Path, patches_path: Path, img_id: str
    ) -> List[Dict[str, Path]]:
        """
        Build paths to all files containing image channels.
        :param base_path: root path containing directories with image channels.
        :param patches_path: path to images patches names to load
        :param img_id: image ID; if specified, load paths for this image only.
        :return: list of dicts containing paths to files with image channles.
        """
        # Get red channel filenames
        if img_id is None:
            red_files = list(base_path.glob("*red/*.TIF"))
        else:
            red_files = list(base_path.glob(f"*red/*{img_id}.TIF"))
        if patches_path is not None:
            patches_names = set(
                np.genfromtxt(
                    patches_path,
                    dtype="str",
                    skip_header=1,
                )
            )
            select_files = []
            for fname in red_files:
                fname_str = str(fname)
                if (
                    fname_str[fname_str.find("patch"): fname_str.find(".TIF")]
                    in patches_names
                ):
                    select_files.append(fname)
            red_files = select_files
        red_files.sort()
        # Get other channels in accordance to the red channel filenames
        return [combine_channel_files(red_file) for red_file in red_files]

    files = build_paths(base_path, patches_path, img_id)
    print(f"Loaded paths for images of { len(files) } samples")

    if shuffle:
        saved_seed = np.random.get_state()
        np.random.seed(42)
        np.random.shuffle(files)
        np.random.set_state(saved_seed)

    if sum(split_ratios) != 1:
        raise RuntimeError("Split ratios don't sum up to one.")

    split_beg = 0
    splits = []
    for ratio in split_ratios:
        split_end = split_beg + math.ceil(ratio * len(files))
        splits.append(files[split_beg:split_end])
        split_beg = split_end

    return splits


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
        :param dim: Tuple with x, y image dimensions.
        :param shuffle: if True shuffles dataset on each epoch end.
        :param with_gt: if True returns y along with x.
        """
        self._batch_size: int = batch_size
        self._dim: Tuple[int, int] = dim
        self._shuffle: bool = shuffle
        self._with_gt: bool = with_gt
        self._balance_snow: bool = balance_snow
        self._balance_classes: bool = balance_classes

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
            self, brightness_thr=0.4, frequency_thr=0.1):
        """
        Returns the pseudo-labels for each patch. Pseudo-label being
        1 if certain percent of pixels in patch are above brightness threshold,
        and 0 otherwise.
        """
        labels = []
        print(len(self._files))
        for file_ in self._files:
            img = self._open_as_array(file_)
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
            gt = self._open_mask(file_)
            clouds_prop = np.count_nonzero(gt) / np.prod(self._dim)
            if clouds_prop > 0.1 and clouds_prop < 0.9:
                labels.append(1)
            else:
                labels.append(0)
        return labels

    def _open_as_array(self, channel_files: Dict[str, Path]) -> np.ndarray:
        """
        Load image as array from given files.
        :param channel_files: Dict with paths to files containing each channel
                              of an image, keyed as 'red', 'green', 'blue',
                              'nir'.
        """
        array_img = np.stack(
            [
                np.array(
                    load_img(channel_files["red"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["green"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["blue"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["nir"], color_mode="grayscale")),
            ],
            axis=2,
        )

        # Return normalized
        return array_img / np.iinfo(array_img.dtype).max

    def _open_mask(self, channel_files: Dict[str, Path]) -> np.ndarray:
        """
        Load ground truth mask as array from given files.
        :param channel_files: Dict with paths to files containing each channel
                              of an image, must contain key 'gt'.
        """
        masks = np.array(load_img(channel_files["gt"], color_mode="grayscale"))
        return np.expand_dims(masks / 255, axis=-1)

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
            x[i] = self._open_as_array(self._files[file_index])
            if self._with_gt:
                y[i] = self._open_mask(self._files[file_index])

        return x, y

    def on_epoch_end(self):
        """
        Triggered after each epoch, if shuffle is randomises file indexing.
        """
        if self._shuffle:
            np.random.shuffle(self._file_indexes)

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self._file_indexes) / self._batch_size))

    def __getitem__(self, index: int):
        """
        Generates one batch of data.
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
    Data generator for Cloud38 clouds segmentation dataset.
    Works with Keras generators.
    """

    def __init__(
        self,
        img_path: Path,
        img_name: str,
        batch_size: int,
        patch_size: int = 384,
        shuffle: bool = True,
    ):
        """
        Prepare generator and init paths to files containing image channels.
        :param img_path: path to the main directory with test scenes.
        :param batch_size: size of generated batches, only one batch is loaded
            to memory at a time.
        :param patch_size: size of the patches.
        :param shuffle: if True shuffles dataset on each epoch end.
        """
        self._batch_size: int = batch_size
        self._patch_size: int = patch_size
        self._shuffle: bool = shuffle

        channel_files = {}
        channel_files["red"] = img_path / img_name / f"{img_name}_B4.TIF"
        channel_files["green"] = img_path / img_name / f"{img_name}_B3.TIF"
        channel_files["blue"] = img_path / img_name / f"{img_name}_B2.TIF"
        channel_files["nir"] = img_path / img_name / f"{img_name}_B5.TIF"
        img = self._open_as_array(channel_files)
        img = pad(img, patch_size)
        self.img_shape = img.shape
        self.patches = rearrange(
            img, "(r dr) (c dc) b -> (r c) dr dc b",
            dr=patch_size, dc=patch_size
        )
        if self._shuffle:
            np.random.shuffle(self.patches)
        del img

    def _open_as_array(self, channel_files: Dict[str, Path]) -> np.ndarray:
        """
        Load image as array from given files. Normalises images on load.
        :param channel_files: Dict with paths to files containing each channel
                              of an image, keyed as 'red', 'green', 'blue',
                              'nir'.
        :return: given image as a single numpy array.
        """
        array_img = np.stack(
            [
                np.array(
                    load_img(channel_files["red"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["green"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["blue"], color_mode="grayscale")),
                np.array(
                    load_img(channel_files["nir"], color_mode="grayscale")),
            ],
            axis=2,
        )

        # Return normalized
        return array_img / np.iinfo(array_img.dtype).max

    def on_epoch_end(self):
        """
        Triggered after each epoch, if shuffle is randomises file indexing.
        """
        if self._shuffle:
            np.random.shuffle(self.patches)

    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self.patches) / self._batch_size))

    def __getitem__(self, index: int):
        """
        Generates one batch of data.
        :return: (x, y) (or (x, None) if with_gt is False) data for one batch,
                 where x is set of RGB + nir images and y is set of
                 corresponding cloud masks.
        """
        return (
            self.patches[index *
                         self._batch_size: (index + 1) * self._batch_size],
            None,
        )


def main():
    """ Demo data loading and present one data sample. """
    base_path = Path("../datasets/clouds/38-Cloud/38-Cloud_training")

    split_names = ("train", "validation", "test")
    splits = load_image_paths(
        base_path=base_path, split_ratios=(0.8, 0.15, 0.05))

    for name, split in zip(split_names, splits):
        dg = DG_38Cloud(split, 16)
        sample_batch_x, sample_batch_y = dg[3]

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(strip_nir(sample_batch_x[0]))
        plt.title(f"Split: { name }\n sample image")

        plt.subplot(1, 3, 2)
        plt.imshow(sample_batch_y[0])
        plt.title(f"Split: { name }\n sample gt mask")

    plt.show()


if __name__ == "__main__":
    main()
