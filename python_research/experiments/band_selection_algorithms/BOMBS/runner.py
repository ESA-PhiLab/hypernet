import os

from tqdm import tqdm

from python_research.experiments.band_selection_algorithms.BOMBS.immune_system_based_model import AntibodyPopulation
from python_research.experiments.band_selection_algorithms.BOMBS.utils import arguments


def main(args):
    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)
    model = AntibodyPopulation(args=args)
    model.initialization()
    for g in tqdm(range(args.G), total=args.Gmax):
        model.update_dominant_population()
        model.serialize_individuals()
        if model.stop_condition(current_generation=g):
            break
        print("Generation: {0}/{1}".format(g + 1, args.Gmax))
        model.show_bands()
        model.serialize_individuals()
        model.active_population_selection()
        model.clone_crossover_mutation(generation_idx=g)
        model.update_antibody_population()
        model.end_generation()
    print("Final {0} selected bands:".format(args.bands_per_antibody))
    model.show_bands()
    model.serialize_individuals()


if __name__ == "__main__":
    main(arguments())
