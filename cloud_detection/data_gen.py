""" Generator based data loader for Cloud38 clouds segmentation dataset. """

import math

import numpy as np
from einops import rearrange
from matplotlib import pyplot as plt
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from typing import Dict, List, Tuple

from utils import overlay_mask, pad

def strip_nir(hyper_img: np.ndarray) -> np.ndarray:
    """
    Strips nir channel so image can be displayed.
    param hyper_img: image with shape (x, y, 4) where fourth channel is nir.
    return: image with shape (x, y, 3) with standard RGB channels.
    """
    return hyper_img[:,:,:3]


def strip_category(mask_img: np.ndarray) -> np.ndarray:
    """
    Strip additional mask dimension, so it can be displayed.
    param mask_img: image mask with shape (x, y, 2).
    return: image mask with shape (x, y), where 1 is cloud.
    """
    return mask_img[:,:,1]


def load_image_paths(base_path: Path,
                     split_ratios: List[float]=[1.0],
                     img_id: str=None) -> List[List[Dict[str, Path]]]:
    """
    Build paths to all files containg image channels. 
    param base_path: root path containing directories with image channels.
    param split_ratios: list containg split ratios, splits should add up to one.
    param img_id: image ID; if specified, load paths for this image only.
    return: list with paths to image files, separated into splits.
        Structured as: list_of_splits[list_of_files['file_channel', Path]]
    """

    def combine_channel_files(red_file: Path) -> Dict[str, Path]:
        """
        Get paths to 'green', 'blue', 'nir' and 'gt' channel files
        based on path to the 'red' channel of the given image.
        param red_file: path to red channel file.
        return: dictionary containing paths to files with each image channel.
        """
        return {
            "red" : red_file,
            "green" : Path(str(red_file).replace("red", "green")),
            "blue" : Path(str(red_file).replace("red", "blue")),
            "nir" : Path(str(red_file).replace("red", "nir")),
            "gt" : Path(str(red_file).replace("red", "gt"))
        }


    def build_paths(base_path: Path, img_id: str) -> List[Dict[str, Path]]: 
        """
        Build paths to all files containg image channels. 
        param base_path: root path containing directories with image channels.
        param img_id: image ID; if specified, load paths for this image only.
        return: list of dicts containing paths to files with image channles.
        """
        # Get red channel filenames
        if img_id is None:
            red_files = list(base_path.glob("*red/*.TIF"))
        else:
            red_files = list(base_path.glob(f"*red/*{img_id}.TIF"))
        # Get other channels in accordance to the red channel filenames
        return [combine_channel_files(red_file) for red_file in red_files]


    files = build_paths(base_path, img_id)
    print(f"Loaded paths for images of { len(files) } samples")

    if sum(split_ratios) != 1:
        raise RuntimeError("Split ratios don't sum up to one.")

    split_beg = 0
    splits = []
    for ratio in split_ratios:
        split_end = split_beg + math.floor(ratio * len(files))
        splits.append(files[split_beg:split_end])
        split_beg=split_end

    return splits


class DataGenerator(keras.utils.Sequence):
    """
    Data generator for Cloud38 clouds segmentation dataset.
    Works with Keras generators.
    """
    def __init__(self,
                 files,
                 batch_size: int,
                 dim: Tuple[int, int]=(384, 384),
                 shuffle: bool=True,
                 with_gt: bool=True
                 ):
        """
        Prepare generator and init paths to files containing image channels.
        param batch_size: size of generated batches, only one batch is loaded
              to memory at a time.
        param dim: Tuple with x, y image dimensions.
        param shuffle: if True shuffles dataset on each epoch end.
        param with_gt: if True returns y along with x.
        """
        self._batch_size: int = batch_size
        self._dim: Tuple[int, int] = dim
        self._shuffle: bool = shuffle
        self._with_gt: bool = with_gt

        self._files = files
        self._file_indexes = np.arange(len(self._files))


    def _open_as_array(self, channel_files: Dict[str, Path]) -> np.ndarray:
        """
        Load image as array from given files.
        param channel_files: Dict with paths to files containing each channel of
            an image, keyed as 'red', 'green', 'blue', 'nir'.
        """
        array_img = np.stack([
            np.array(load_img(channel_files['red'], color_mode="grayscale")),
            np.array(load_img(channel_files['green'], color_mode="grayscale")),
            np.array(load_img(channel_files['blue'], color_mode="grayscale")),
            np.array(load_img(channel_files['nir'], color_mode="grayscale"))
            ], axis=2)

        # Return normalized
        return (array_img / np.iinfo(array_img.dtype).max)


    def _open_mask(self, channel_files: Dict [str, Path]) -> np.ndarray:
        """
        Load ground truth mask as array from given files.
        :param channel_files: Dict with paths to files containing each channel of
            an image, must contain key 'gt'.
        """
        masks = np.array(load_img(channel_files['gt'], color_mode="grayscale"))
        return np.expand_dims(masks/255, axis=-1)


    def _data_generation(self, file_indexes_to_gen: np.arange) -> Tuple:
        """
        Generates data containing batch_size samples.
        param file_indexes_to_gen: Sequence of indexes of files from which images
            should be loaded.
        return: (x, y) (or (x, None) if with_gt is False) data for one batch, where x is
            set of RGB + nir images and y is set of corresponding cloud masks.
        """
        x = np.empty((len(file_indexes_to_gen), *self._dim, 4))
        if self._with_gt == True:
            y = np.empty((len(file_indexes_to_gen), *self._dim, 1))
        else:
            y = None

        for i, file_index in enumerate(file_indexes_to_gen):
            x[i] = self._open_as_array(self._files[file_index])
            if self._with_gt == True:
                y[i] = self._open_mask(self._files[file_index])

        return x, y


    def on_epoch_end(self):
        """ Triggered after each epoch, if shuffle is randomises file indexing. """
        if self._shuffle == True:
            np.random.shuffle(self._file_indexes)


    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self._file_indexes) / self._batch_size))


    def __getitem__(self, index: int):
        """
        Generates one batch of data.
        return: (x, y) (or (x, None) if with_gt is False) data for one batch, where x is
            set of RGB + nir images and y is set of corresponding cloud masks.
        """ 
        # Generate indexes of the batch
        indexes_in_batch = self._file_indexes[index*self._batch_size:(index+1)*self._batch_size]

        # Generate data
        return self._data_generation(indexes_in_batch)


class DataGenerator_L8CCA(keras.utils.Sequence):
    """
    Data generator for Cloud38 clouds segmentation dataset.
    Works with Keras generators.
    """
    def __init__(self,
                 img_path: Path,
                 img_name: str,
                 batch_size: int,
                 patch_size: int=384,
                 shuffle: bool=True
                 ):
        """
        Prepare generator and init paths to files containing image channels.
        param batch_size: size of generated batches, only one batch is loaded
              to memory at a time.
        param patch_size: size of the patches.
        param shuffle: if True shuffles dataset on each epoch end.
        """
        self._batch_size: int = batch_size
        self._patch_size: int = patch_size
        self._shuffle: bool = shuffle

        channel_files = {}
        channel_files['red'] = img_path / img_name / f'{img_name}_B4.TIF'
        channel_files['green'] = img_path / img_name / f'{img_name}_B3.TIF'
        channel_files['blue'] = img_path / img_name / f'{img_name}_B2.TIF'
        channel_files['nir'] = img_path / img_name / f'{img_name}_B5.TIF'
        img = self._open_as_array(channel_files)
        img = pad(img, patch_size)
        self.img_shape = img.shape
        self.patches = rearrange(img, '(r dr) (c dc) b -> (r c) dr dc b',
                                 dr=patch_size, dc=patch_size)
        del img


    def _open_as_array(self, channel_files: Dict[str, Path]) -> np.ndarray:
        """
        Load image as array from given files.
        param channel_files: Dict with paths to files containing each channel of
            an image, keyed as 'red', 'green', 'blue', 'nir'.
        """
        array_img = np.stack([
            np.array(load_img(channel_files['red'], color_mode="grayscale")),
            np.array(load_img(channel_files['green'], color_mode="grayscale")),
            np.array(load_img(channel_files['blue'], color_mode="grayscale")),
            np.array(load_img(channel_files['nir'], color_mode="grayscale"))
            ], axis=2)

        # Return normalized
        return (array_img / np.iinfo(array_img.dtype).max)


    def on_epoch_end(self):
        """ Triggered after each epoch, if shuffle is randomises file indexing. """
        if self._shuffle == True:
            np.random.shuffle(self.patches)


    def __len__(self):
        """ Denotes the number of batches per epoch. """
        return int(np.ceil(len(self.patches) / self._batch_size))


    def __getitem__(self, index: int):
        """
        Generates one batch of data.
        return: (x, y) (or (x, None) if with_gt is False) data for one batch, where x is
            set of RGB + nir images and y is set of corresponding cloud masks.
        """
        return self.patches[index*self._batch_size:(index+1)*self._batch_size], None


def main():
    """ Demo data loading and present one data sample. """
    base_path = Path("../datasets/clouds/38-Cloud/38-Cloud_training")

    split_names = ("train", "validation", "test")
    splits = load_image_paths(base_path, (0.1, 0.2, 0.7))
    
    for name, split in zip(split_names, splits):
        dg = DataGenerator(split, 16)
        sample_batch_x, sample_batch_y = dg[3]

        plt.figure()

        plt.subplot(1, 3, 1)
        plt.imshow(strip_nir(sample_batch_x[0]))
        plt.title(f"Split: { name }\n sample image")

        plt.subplot(1, 3, 2)
        plt.imshow(sample_batch_y[0])
        plt.title(f"Split: { name }\n sample gt mask")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay_mask(strip_nir(sample_batch_x[0]), sample_batch_y[0]))
        plt.title(f"Split: { name }\n sample gt mask overlay")

    plt.show()


if __name__ == '__main__':
    main()
