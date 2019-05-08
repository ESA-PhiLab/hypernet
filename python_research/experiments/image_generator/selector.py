import os
from random import shuffle

import gdal
import numpy as np
import osr
from scipy.io import loadmat


def load_data(path: str) -> np.ndarray:
    """
    Load data for image generation.

    :param path: Path to the dataset.
    :return: Loaded dataset.
    """
    data = None
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    return data.astype(float)


def get_images(data_name: str, path: str, bands_txt: str, dest_path: str, do_shuffle: bool) -> None:
    """
    Generate images based on selected bands.

    :param data_name: Name of the data.
    :param path: Path to data.
    :param bands_txt: Path to .txt file containing selected bands.
    :param dest_path: Path to destination directory.
    :param do_shuffle: Boolean indicating whether to shuffle spectral bands.
    :return: None.
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    os.chdir(dest_path)
    data = load_data(path)
    with open(bands_txt) as f:
        content = f.readlines()
        content = np.asarray([x.rstrip("\n") for x in content], dtype=int)
    if do_shuffle:
        shuffle(content)
    data *= (255.0 / data.max())
    for i in range(0, len(content) - 2):
        image = data[..., content[slice(i, i + 3)]]
        x, y, z = image.shape
        dst_ds = gdal.GetDriverByName("GTiff").Create("{0}_{1}_.tif".format(
            data_name, content[slice(i, i + 3)]), y, x, z, gdal.GDT_Byte)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(0)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).WriteArray(image[..., 0])
        dst_ds.GetRasterBand(2).WriteArray(image[..., 1])
        dst_ds.GetRasterBand(3).WriteArray(image[..., 2])
        dst_ds.FlushCache()
