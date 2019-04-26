from python_research.experiments.band_selection_algorithms.BOMBS.utils import prep_bands
from python_research.experiments.band_selection_algorithms.utils import *


class Antibody(object):
    def __init__(self, selected_bands: np.ndarray, bands_ids: list, data_path: str, ref_map_path: str):
        """
        Initialize all instance variables of the antibody.

        :param selected_bands: Numpy array containing selected bands.
        :param bands_ids: List of selected bands.
        :param data_path: Path to the data file.
        :param ref_map_path: Path to the reference map file.
        """
        self.grey_scale_hist = prep_bands(selected_bands=selected_bands)
        self.bands_ids = bands_ids
        self.entropy_fitness = None
        self.distance_fitness = None
        self.dominant_fitness = None
        self.K = selected_bands.shape[SPECTRAL_AXIS]
        self.dominant_fitness_check = lambda: np.max([self.entropy_fitness, self.distance_fitness])
        self.data_path = data_path
        self.ref_map_path = ref_map_path
        self.Sp = []
        self.L = len(np.unique(self.bands_ids).tolist())

    def __add__(self, other):
        """
        Add method used in "whole arithmetic crossover".

        :param other: Antibody individual.
        :return: Returns antibody individual.
        """
        if isinstance(other, int):
            if other == 0:
                return self

    def __radd__(self, other):
        """
        Add method used in "whole arithmetic crossover".

        :param other: Antibody individual.
        :return: Returns antibody individual.
        """
        if isinstance(other, int):
            if other == 0:
                return self

    def __mul__(self, other):
        """
        Multiplication method used in "whole arithmetic crossover."

        :param other: Integer value in range [0;1]
        :return: Antibody object or zero.
        """
        if isinstance(other, int):
            if other == 0:
                return 0
            if other == 1:
                return self

    def __rmul__(self, other):
        """
         Multiplication method used in "whole arithmetic crossover."

         :param other: Integer value in range [0;1]
         :return: Antibody object or zero.
         """
        if isinstance(other, int):
            if other == 0:
                return 0
            if other == 1:
                return self

    def refresh_bands(self, data):
        """
        Set unique bands for each antigen - it will lower the value
        of objective functions for individuals which have repeating bands ids.
        """
        selected_bands = data[..., np.unique(self.bands_ids)]
        self.grey_scale_hist = prep_bands(selected_bands=selected_bands)
        self.L = len(np.unique(self.bands_ids).tolist())

    def calculate_fitness(self):
        """
        Calculate fitness of the antibody.
        """
        self.entropy_fitness = self.calculate_entropy()
        self.distance_fitness = self.calculate_distance()
        self.dominant_fitness = self.dominant_fitness_check()

    def calculate_entropy(self):
        """
        For measuring the information or uncertainty of selected bands a Entropy function is used.
        High entropy means that a random variable is informative and uncertain. Low entropy indicates,
        that a random variable is not that random and not that informative.
        (Zero probabilities are not taken into consideration.)
        """
        entropy_sum = 0
        for i in range(len(self.grey_scale_hist)):
            entropy_sum += -np.sum(self.grey_scale_hist[i] * np.ma.log2(self.grey_scale_hist[i]))
        return entropy_sum / self.K

    def calculate_distance(self):
        """
        Cross Entropy is adopted as the distance criterion between selected bands in the antibody.
        """
        distances = 0
        for i in range(self.grey_scale_hist.__len__()):
            for j in range(i + 1, self.grey_scale_hist.__len__()):
                distances += -np.sum(self.grey_scale_hist[i] * np.ma.log2(self.grey_scale_hist[j])) + \
                             -np.sum(self.grey_scale_hist[j] * np.ma.log2(self.grey_scale_hist[i]))
        distances = (2 / (self.K * (self.K - 1))) * distances
        return distances

    def clear_individual(self):
        """
        Clear fields of each individual.
        """
        self.entropy_fitness = None
        self.distance_fitness = None
        self.dominant_fitness = None
        self.Sp = []
