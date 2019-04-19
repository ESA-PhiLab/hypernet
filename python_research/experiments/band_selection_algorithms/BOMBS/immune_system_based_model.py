import copy
import math
import os
import random

import numpy as np

from python_research.experiments.band_selection_algorithms.BOMBS.antibody_individual import Antibody
from python_research.experiments.band_selection_algorithms.BOMBS.utils import load_data, calculate_crowding_distances
from python_research.experiments.band_selection_algorithms.utils import SPECTRAL_AXIS


class AntibodyPopulation(object):
    def __init__(self, args):
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
        self.b = 1  # Degree of non-uniformity system parameter.

    def initialization(self):
        """
        Step 1. Initialization.
        Generate an initial antibody population P_zero, with a size Nd.
        """
        bands = self.randomize_bands(args=self.args)
        [self.P.append(Antibody(selected_bands=selected_bands, bands_ids=bands_ids, data_path=self.args.data_path,
                                ref_map_path=self.args.ref_map_path)) for selected_bands, bands_ids in bands]

    def update_dominant_population(self):
        """
        Step 2. Update Dominant Population.
        Calculate fitness for all antigens in the population P.

        Identify all dominant antibodies in P by NONDOMINATED SORTING.
        Select the dominant antibodies into a temporary dominant population TD.

        If size of temporary population is greater then size of dominant population
        perform crowding distance procedure and select antibodies.
        """
        for antigen in self.P:
            antigen.calculate_fitness()
        if len(self.P) == self.args.Nd:
            self.D = self.P
        if len(self.P) > self.args.Nd:
            self.TD = self.nondominated_sort()[:self.args.TD_size]
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

    def _copy(self, bands_ids: list, deep=None) -> Antibody:
        """
        Copy selected bands from given antibody and create its clone.

        :param bands_ids: Passed list of selected bands.
        :param deep: Flag determining the kind of copy procedure.
        :return: Cloned antibody.
        """
        antibody_copy = Antibody(selected_bands=load_data(self.args.data_path,
                                                          self.args.ref_map_path)[..., np.unique(bands_ids)],
                                 bands_ids=bands_ids,
                                 data_path=self.args.data_path, ref_map_path=self.args.ref_map_path)
        if deep is not None:
            antibody_copy.entropy_fitness = copy.deepcopy(deep.entropy_fitness)
            antibody_copy.distance_fitness = copy.deepcopy(deep.distance_fitness)
            antibody_copy.dominant_fitness = copy.deepcopy(deep.dominant_fitness)
        return antibody_copy

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
        Step 6: Update Antibody Population:
        """
        self.P = self.C_prime + self.D

    def heterogeneous_mutation(self, generation_idx: int):
        """
        Heterogeneous mutation.
        Each band of each individual has a probability of adding a integer from interval [0, z],
        thus, changing the chosen band index to another one.

        :param generation_idx: Index of generation.
        """
        assert self.u_max is not None, 'Value of maximum number is not assigned.'
        data = load_data(self.args.data_path, self.args.ref_map_path)
        for individual in range(len(self.C_prime)):
            alpha = random.uniform(0, 1)
            for i in range(self.C_prime[individual].K):
                p = self.calculate_p(c_ri=self.C_prime[individual].bands_ids[i])
                rand = random.random()
                if rand < p:
                    z = self.u_max - self.C_prime[individual].bands_ids[i]
                    delta = self.delta_tz(z=z, generation_idx=generation_idx, alpha=alpha)
                    self.C_prime[individual].bands_ids[i] += delta
                elif rand > p:
                    z = self.C_prime[individual].bands_ids[i] - self.u_min
                    delta = self.delta_tz(z=z, generation_idx=generation_idx, alpha=alpha)
                    self.C_prime[individual].bands_ids[i] -= delta
            self.C_prime[individual].refresh_bands(data)

    def whole_arithmetic_crossover(self):
        """
        Whole arithmetic crossover.
        """
        while len(self.C_prime) < self.args.Nc:
            a_i, c_j = np.random.choice(self.A, size=1)[0], np.random.choice(self.C, size=1)[0]
            gamma = np.random.randint(low=0, high=2, dtype=int)
            antibody = a_i * gamma + c_j * (1 - gamma)
            antibody = self._copy(antibody.bands_ids.copy())
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
        max_i = np.argmax([i.dominant_fitness for i in self.D]).astype(int)
        print('Bands of the best individual in the whole population: {}'.format(np.sort(self.D[max_i].bands_ids)))
        print('Entropy: {}'.format(self.D[max_i].entropy_fitness),
              'Distance: {}'.format(self.D[max_i].distance_fitness))

    def end_generation(self):
        """
        Clear memory after generation is over.
        """
        self.A.clear(), self.D.clear(), self.TD.clear(), self.C.clear(), self.C_prime.clear()
        map(lambda obj: obj.clear_individual, self.P)

    def calculate_p(self, c_ri):
        return (int(c_ri) - self.u_min) / (self.u_max - self.u_min)

    def delta_tz(self, z, generation_idx, alpha):
        t_prime = (1 - (generation_idx / self.args.Gmax)) ** self.b
        t = (1 - alpha * t_prime)
        delta_tz = int(z * t)
        return delta_tz

    def serialize_individuals(self):
        """
        Save results of the best individual from the population.
        """
        max_i = np.argmax([i.dominant_fitness for i in self.D]).astype(int)
        np.savetxt(os.path.join(self.args.dest_path, 'best_individual_bands'),
                   np.sort(np.unique(self.D[max_i].bands_ids)), fmt='%d')
        np.savetxt(os.path.join(self.args.dest_path, 'best_individual_entropy'),
                   np.sort(np.unique(self.D[max_i].entropy_fitness)), fmt='%5.5f')
        np.savetxt(os.path.join(self.args.dest_path, 'best_individual_distance'),
                   np.sort(np.unique(self.D[max_i].distance_fitness)), fmt='%5.5f')

    def nondominated_sort(self) -> list:
        """
        Fast - non - dominated - sort.
        https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf
        """
        non_dominated_frotns = []
        selected_fronts = []
        for p in self.P:
            p.n = 0
            for q in self.P:
                if p.dominant_fitness > q.dominant_fitness:
                    p.Sp.append(q)
                elif p.dominant_fitness < q.dominant_fitness:
                    p.n += 1
            if p.n == 0:
                selected_fronts.append(p)
        non_dominated_frotns.append(selected_fronts)
        i = 0
        while len(non_dominated_frotns[i]) != 0:
            h = []
            for p in non_dominated_frotns[i]:
                for q in p.Sp:
                    q.n -= 1
                    if q.n == 0:
                        h.append(q)
            i += 1
            non_dominated_frotns.append(h)
        sorted_list = [antibody for sub_fronts in non_dominated_frotns for antibody in sub_fronts]
        return sorted_list

    def stop_condition(self, current_generation: int) -> bool:
        """
        Step 3: Termination check.

        :param current_generation: Index of current generation.
        """
        if current_generation >= self.args.Gmax:
            print('Bail...')
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
        bands_number = args.bands_per_antibody
        for i in range(args.P_init_size):
            chosen_bands = np.random.choice(a=self.u_max, size=bands_number, replace=False)
            population.append([data[..., chosen_bands], chosen_bands])
        return population
