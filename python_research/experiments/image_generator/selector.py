from random import shuffle

import gdal
import numpy as np
import osr
from scipy.io import loadmat

CONST_SPECTRAL_AXIS = -1


def load_data(path):
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


def get_images(path_to_txt: str, path: str, data_name: str, rand=False):
    """
    Generate images that compose of three of selected bands by attention mechanism.
    :param path_to_txt: The path to the file with the largest number of bands selected by attention mechanism.
    :param path: Path to the data file.
    :param data_name: Name of the data set.
    :param rand: Is True if shuffle bands indices.
    pavia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 100, 101, 102]
    salinas = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
              57, 58, 59, 60, 105, 106, 107, 108, 167, 168, 169, 170, 171, 172]
    """
    data = load_data(path)
    with open(path_to_txt) as f:
        content = f.readlines()
        content = np.asarray([x.rstrip('\n') for x in content], dtype=int)
    if rand:
        shuffle(content)
    data = (data - data.min()) * (255 / (data.max() - data.min()))
    for i in range(0, len(content) - 2):
        image = data[..., content[slice(i, i + 3)]]
        x, y, z = image.shape
        dst_ds = gdal.GetDriverByName('GTiff').Create('{0}_{1}_.tif'.format(
            data_name, content[slice(i, i + 3)]), y, x, z, gdal.GDT_Byte)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(0)
        dst_ds.SetProjection(srs.ExportToWkt())
        dst_ds.GetRasterBand(1).WriteArray(image[..., 0])
        dst_ds.GetRasterBand(2).WriteArray(image[..., 1])
        dst_ds.GetRasterBand(3).WriteArray(image[..., 2])
        dst_ds.FlushCache()
