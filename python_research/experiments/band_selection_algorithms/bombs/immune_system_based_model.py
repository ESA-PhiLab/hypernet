import math
import os
import random
from math import pow

import numpy as np

from python_research.experiments.band_selection_algorithms.bombs.antibody_individual import Antibody
from python_research.experiments.band_selection_algorithms.bombs.utils import load_data, calculate_crowding_distances, \
    Arguments
from python_research.experiments.band_selection_algorithms.utils import SPECTRAL_AXIS


class AntibodyPopulation(object):
    def __init__(self, args: Arguments):
        """
        Initialize all fields of the class.
        """
        self.args = args
        # Population lists:
        self.P = []  # Initial antibody population.
        self.A = []  # Active antibody population.
        self.D = []  # Dominant antibody population.
        self.C = []  # Clone antibody population.
        self.C_prime = []  # Updated clone antibody population.
        self.TD = []  # Temporary dominant antibody population.
        # Parameters:
        self.u_min = 1
        self.u_max = None
        self.b = 1  # Degree of non-uniformity, system parameter.

    def initialization(self):
        """
        Step 1. Initialization.
        Generate an initial antibody population P_zero randomly, with a size of Nd.
        """
        bands = self.randomize_bands(args=self.args)
        [self.P.append(Antibody(selected_bands=selected_bands, band_indexes=band_indexes, data_path=self.args.data_path,
                                ref_map_path=self.args.ref_map_path)) for selected_bands, band_indexes in bands]

    def update_dominant_population(self):
        """
        Step 2. Update Dominant Population.
        Calculate fitness for all antibodies in the population P.

        Identify all dominant antibodies in P by NONDOMINATED SORTING.
        Select the dominant antibodies into a temporary dominant population TD.

        If size of temporary population is greater then size of dominant population
        perform crowding distance procedure and select antibodies.
        """
        for antibody in self.P:
            antibody.calculate_objective_functions()
        self.P = self.nondominated_sort()
        if self.P.__len__() == self.args.Nd:
            self.D = self.P
        else:
            self.TD = self.P[:self.args.TD_size]
            if self.args.TD_size > self.args.Nd:
                chosen_antibodies = calculate_crowding_distances(list_of_antibodies=self.TD)
                chosen_antibodies = np.argsort(-chosen_antibodies).tolist()[:self.args.Nd]
                [self.D.append(self.TD[antibody]) for antibody in chosen_antibodies]
            if self.args.TD_size == self.args.Nd:
                self.D = self.TD

    def active_population_selection(self):
        """
        Step 4. Active population selection.
        """
        if len(self.D) > self.args.Na:
            chosen_antibodies = calculate_crowding_distances(list_of_antibodies=self.D)
            chosen_antibodies = np.argsort(-chosen_antibodies).tolist()[:self.args.Na]
            for antibody in chosen_antibodies:
                antibody_ = self.D[antibody]
                self.A.append(antibody_)
        else:
            self.A = self.D

    def _copy(self, band_indexes: list) -> Antibody:
        """
        Copy selected bands from given antibody and create its clone based on the selected bands set.

        :param band_indexes: List of selected bands.
        :return: Cloned antibody.
        """
        return Antibody(selected_bands=load_data(self.args.data_path,
                                                 self.args.ref_map_path)[..., np.unique(band_indexes)],
                        band_indexes=band_indexes,
                        data_path=self.args.data_path, ref_map_path=self.args.ref_map_path)

    def clone_crossover_mutation(self, generation_idx: int):
        """
        Step 5. Clone, crossover, mutation.

        :param generation_idx: Index of generation.
        """
        self.proportional_cloning()
        self.whole_arithmetic_crossover()
        self.heterogeneous_mutation(generation_idx)

    def update_antibody_population(self):
        """
        Step 6: Update Antibody Population.
        """
        self.P = self.C_prime + self.D

    def heterogeneous_mutation(self, generation_idx: int):
        """
        Heterogeneous mutation.
        Each band of each individual has a probability of adding a integer from interval [0, z],
        thus, changing the chosen band index to another one.

        :param generation_idx: Index of generation.
        """
        assert self.u_max is not None, "Value of maximum number of bands," \
                                       "i.e. range of the mutation scope is not assigned."
        data = load_data(self.args.data_path, self.args.ref_map_path)
        for individual_index in range(self.C_prime.__len__()):
            alpha = random.uniform(0, 1)
            for band_index in range(self.C_prime[individual_index].designed_band_size):
                mutation_prob = self.calculate_mutation_prob(
                    c_ri=int(self.C_prime[individual_index].band_indexes[band_index]))
                rand = random.uniform(0, 1)
                if rand < mutation_prob:
                    z_parameter = self.u_max - self.C_prime[individual_index].band_indexes[band_index]
                    mutation_scope = self.calculate_mutation_scope(z_parameter=z_parameter,
                                                                   generation_idx=generation_idx,
                                                                   alpha=alpha)
                    self.C_prime[individual_index].band_indexes[band_index] += mutation_scope
                elif rand < (1 - mutation_prob):
                    z_parameter = self.C_prime[individual_index].band_indexes[band_index] - self.u_min
                    mutation_scope = self.calculate_mutation_scope(z_parameter=z_parameter,
                                                                   generation_idx=generation_idx,
                                                                   alpha=alpha)
                    self.C_prime[individual_index].band_indexes[band_index] -= mutation_scope
            self.C_prime[individual_index].refresh_bands(data)

    def whole_arithmetic_crossover(self):
        """
        Whole arithmetic crossover.
        """
        while len(self.C_prime) < self.args.Nc:
            a_i, c_j = np.random.choice(self.A, size=1)[0], np.random.choice(self.C, size=1)[0]
            gamma = np.random.randint(low=0, high=2, dtype=int)
            antibody = a_i * gamma + c_j * (1 - gamma)
            antibody = self._copy(antibody.band_indexes.copy())
            self.C_prime.append(antibody)

    def proportional_cloning(self):
        """
        Perform "Proportional Cloning" on the active population.
        """
        active_crowding_distances = calculate_crowding_distances(list_of_antibodies=self.A)
        cloning_times = [int(math.ceil(self.args.Nc * (crowding_value / sum(active_crowding_distances))))
                         for crowding_value in active_crowding_distances]
        for i, individual_cloning_times in enumerate(cloning_times):
            self.C.extend([self.A[i]] * individual_cloning_times)
        self.C = [i for i in self.C[:self.args.Nc]]

    def show_bands(self):
        """
        Show results concerning given generation.
        """
        max_ = np.argmax([antibody.dominant_fitness for antibody in self.D]).astype(int)
        print("Bands of the best individual in the whole population: {}".format(np.sort(self.D[max_].band_indexes)))
        print("Entropy: {}".format(self.D[max_].entropy_fitness),
              "Distance: {}".format(self.D[max_].distance_fitness))

    def end_generation(self):
        """
        Clear memory after each generation.
        """
        self.A.clear(), self.D.clear(), self.TD.clear(), self.C.clear(), self.C_prime.clear()
        map(lambda obj: obj.clear_individual, self.P)

    def calculate_mutation_prob(self, c_ri: int) -> float:
        """
        Calculate probability "p" for heterogeneous mutation.

        :param c_ri: Number of selected band.
        :return: Probability of heterogeneous mutation.
        """
        return float((c_ri - self.u_min) / (self.u_max - self.u_min))

    def calculate_mutation_scope(self, z_parameter: int, generation_idx: int, alpha: float) -> int:
        """
        Calculate mutation scope for each individual antibody.

        :param z_parameter: Z parameter designed for mutation.
        :param generation_idx: Index of the actual generation.
        :param alpha: Alpha parameter-random value with a range (0, 1).
        :return: Mutation scope as the integer value.
        """
        t_parameter = pow(1 - (generation_idx / self.args.Gmax), self.b)
        return int(z_parameter * (1 - pow(alpha, t_parameter)))

    def serialize_individuals(self):
        """
        Save artifacts of the best individual from the population.
        """
        max_i = np.argmax([i.dominant_fitness for i in self.D]).astype(int)
        np.savetxt(os.path.join(self.args.dest_path, "best_individual_bands"),
                   np.sort(np.unique(self.D[max_i].band_indexes)), fmt="%d")
        np.savetxt(os.path.join(self.args.dest_path, "best_individual_entropy"),
                   np.sort(np.unique(self.D[max_i].entropy_fitness)), fmt="%5.5f")
        np.savetxt(os.path.join(self.args.dest_path, "best_individual_distance"),
                   np.sort(np.unique(self.D[max_i].distance_fitness)), fmt="%5.5f")

    def nondominated_sort(self) -> list:
        """
        Nondominated sorting algorithm.
        """
        non_dominated_frotns = []
        selected_fronts = []
        for p_antibody in self.P:
            p_antibody.n_sorting_index = 0
            for q_antibody in self.P:
                if p_antibody.dominant_fitness > q_antibody.dominant_fitness:
                    p_antibody.sp_antibody_set.append(q_antibody)
                elif p_antibody.dominant_fitness < q_antibody.dominant_fitness:
                    p_antibody.n_sorting_index += 1
            if p_antibody.n_sorting_index == 0:
                selected_fronts.append(p_antibody)
        non_dominated_frotns.append(selected_fronts)
        iterator = 0
        while len(non_dominated_frotns[iterator]) != 0:
            temp_h_list = []
            for p_antibody in non_dominated_frotns[iterator]:
                for q_antibody in p_antibody.sp_antibody_set:
                    q_antibody.n_sorting_index -= 1
                    if q_antibody.n_sorting_index == 0:
                        temp_h_list.append(q_antibody)
            iterator += 1
            non_dominated_frotns.append(temp_h_list)
        sorted_list = [antibody for sub_fronts in non_dominated_frotns for antibody in sub_fronts]
        return sorted_list

    def stop_condition(self, current_generation: int) -> bool:
        """
        Step 3: Termination check.

        :param current_generation: Index of current generation.
        """
        if current_generation >= self.args.Gmax:
            print("Bail...")
            return True
        else:
            return False

    def randomize_bands(self, args) -> list:
        """
        Generate bands for each antibody.

        :param args: Parsed arguments.
        :return: List of randomized hyperspectral data blocks.
        """
        population = []
        data = load_data(self.args.data_path, self.args.ref_map_path)
        self.u_max = data.shape[SPECTRAL_AXIS]
        for i in range(args.P_init_size):
            chosen_bands = np.random.choice(a=self.u_max, size=args.bands_per_antibody, replace=False)
            population.append([data[..., chosen_bands], chosen_bands])
        return population
