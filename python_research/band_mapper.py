import numpy as np

SAMPLES = 0


class BandMapper:
    """
    Class implementing bands mapping to a specified number. Bands can be mapped
    in different ways (specified as method parameter):
    - average - values of pixels are average across a chunk of bands which size
    is dependent on resulting number of bands.
    - min - min value of pixel across a chunk of bands is taken.
    - max - max value of pixel across a chunk of bands is taken.
    Chunks of bands to merge are calculated with numpy.array_split (please
    refer to Numpy documentation for more details).
    """
    def map(self, data: np.ndarray, resulting_bands_count: int,
            method="average") -> np.ndarray:
        if method == "average":
            return self._map_average(data, resulting_bands_count)
        if method == "min":
            return self._map_min(data, resulting_bands_count)
        if method == "max":
            return self._map_max(data, resulting_bands_count)

    @staticmethod
    def _map_average(data: np.ndarray, resulting_bands_count: int) \
            -> np.ndarray:
        sub_arrays = np.array_split(data, resulting_bands_count, axis=1)
        mapped_data = np.zeros((data.shape[SAMPLES], resulting_bands_count))
        for index, sub_array in enumerate(sub_arrays):
            mapped_sub_array = np.average(sub_array, axis=1)
            mapped_data[..., index] = mapped_sub_array
        return mapped_data

    @staticmethod
    def _map_max(data: np.ndarray, resulting_bands_count: int) \
            -> np.ndarray:
        sub_arrays = np.array_split(data, resulting_bands_count, axis=1)
        mapped_data = np.zeros((data.shape[SAMPLES], resulting_bands_count))
        for index, sub_array in enumerate(sub_arrays):
            mapped_sub_array = np.max(sub_array, axis=1)
            mapped_data[..., index] = mapped_sub_array
        return mapped_data

    @staticmethod
    def _map_min(data: np.ndarray, resulting_bands_count: int) \
            -> np.ndarray:
        sub_arrays = np.array_split(data, resulting_bands_count, axis=1)
        mapped_data = np.zeros((data.shape[SAMPLES], resulting_bands_count))
        for index, sub_array in enumerate(sub_arrays):
            mapped_sub_array = np.min(sub_array, axis=1)
            mapped_data[..., index] = mapped_sub_array
        return mapped_data
