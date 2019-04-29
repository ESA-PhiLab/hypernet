import numpy as np


class SpectralBand(object):
    def __init__(self, histogram: np.ndarray, joint_histogram: np.ndarray, band_index: int):
        """
        Spectral band class initializer.

        :param histogram: Normalized band histogram.
        :param joint_histogram: Joint histogram between specific band and ground truth map.
        :param band_index: Index of passed band.
        """
        self.histogram = histogram
        self.joint_histogram = joint_histogram
        self.band_index = band_index
        self.mutual_information = None
