from python_research.experiments.band_selection_algorithms.bombs.utils import prep_bands
from python_research.experiments.band_selection_algorithms.utils import *


class Antibody(object):
    def __init__(self, selected_bands: np.ndarray, band_indexes: list, data_path: str, ref_map_path: str):
        """
        Initialize all instance variables of the antibody.

        :param selected_bands: Numpy array containing selected bands.
        :param band_indexes: List of selected bands.
        :param data_path: Path to the data file.
        :param ref_map_path: Path to the ground truth map.
        """
        self.grey_level_histograms = prep_bands(selected_bands=selected_bands)
        self.band_indexes = band_indexes
        self.entropy_fitness = None
        self.distance_fitness = None
        self.dominant_fitness = None
        self.designed_band_size = selected_bands.shape[SPECTRAL_AXIS]
        self.dominant_fitness_check = lambda: np.max([self.entropy_fitness, self.distance_fitness])
        self.data_path = data_path
        self.ref_map_path = ref_map_path
        self.sp_antibody_set = []
        self.n_sorting_index = None
        self.unique_band_size = len(np.unique(self.band_indexes).tolist())

    def __add__(self, antibody):
        """
        Add method used in "whole arithmetic crossover" step.

        :param antibody: Antibody individual.
        :return: Antibody individual.
        """
        if isinstance(antibody, int):
            if antibody == 0:
                return self

    def __radd__(self, antibody):
        """
        Add method used in "whole arithmetic crossover" step.

        :param antibody: Antibody individual.
        :return: Antibody individual.
        """
        if isinstance(antibody, int):
            if antibody == 0:
                return self

    def __mul__(self, antibody):
        """
        Multiplication method used in "whole arithmetic crossover" step.

        :param antibody: Integer value in range [0;1].
        :return: Antibody object or zero.
        """
        if isinstance(antibody, int):
            if antibody == 0:
                return 0
            else:
                return self

    def __rmul__(self, antibody):
        """
         Multiplication method used in "whole arithmetic crossover" step.

         :param antibody: Integer value in range [0;1]
         :return: Antibody object or zero.
         """
        if isinstance(antibody, int):
            if antibody == 0:
                return 0
            else:
                return self

    def refresh_bands(self, data: np.ndarray):
        """
        Set unique bands for each antibody, this process will lower the value
        of objective functions for individuals, which have repeating bands indexes.
        """
        selected_bands = data[..., np.unique(self.band_indexes)]
        self.grey_level_histograms = prep_bands(selected_bands=selected_bands)
        self.unique_band_size = len(np.unique(self.band_indexes).tolist())

    def calculate_objective_functions(self):
        """
        Calculate values of objective functions of the antibodies.
        """
        self.entropy_fitness = self.calculate_entropy()
        self.distance_fitness = self.calculate_distance()
        self.dominant_fitness = self.dominant_fitness_check()

    def calculate_entropy(self) -> float:
        """
        For measuring the information or uncertainty of selected bands a entropy - based objective function is used.
        High entropy means that a random variable is informative and uncertain, whereas low entropy indicates,
        that a random variable is not that random and not that informative.
        (Zero probabilities are not taken into consideration.)
        """
        entropy_sum = 0
        for i in range(len(self.grey_level_histograms)):
            entropy_sum += -np.sum(self.grey_level_histograms[i] * np.ma.log2(self.grey_level_histograms[i]))
        return float(entropy_sum / self.designed_band_size)

    def calculate_distance(self) -> float:
        """
        Cross Entropy is adopted as the distance criterion between selected bands in the antibody.
        """
        distances = 0
        for i in range(self.grey_level_histograms.__len__()):
            for j in range(i + 1, self.grey_level_histograms.__len__()):
                distances += -np.sum(self.grey_level_histograms[i] * np.ma.log2(self.grey_level_histograms[j])) + \
                             -np.sum(self.grey_level_histograms[j] * np.ma.log2(self.grey_level_histograms[i]))
        return float((2 / (self.designed_band_size * (self.designed_band_size - 1))) * distances)

    def clear_individual(self):
        """
        Clear fields of each individual.
        """
        self.entropy_fitness = None
        self.distance_fitness = None
        self.dominant_fitness = None
        self.sp_antibody_set = []
