import argparse

from python_research.experiments.band_selection_algorithms.BOMBS.immune_system_based_model import AntibodyPopulation
from python_research.experiments.band_selection_algorithms.BOMBS.utils import arguments


def arg_parser() -> argparse.Namespace:
    """
    Parse arguments for BOMBS band selection algorithm.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for band selection based on improved classification map.')
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--ref_map_path', dest='ref_map_path', type=str)
    parser.add_argument('--dest_path', dest='dest_path', type=str)
    parser.add_argument('--neighborhood_size', dest='r', type=int, default=5)
    parser.add_argument('--training_patch', dest='training_patch', type=float)
    parser.add_argument('--bands_num', dest='bands_num', type=int)
    return parser.parse_args()


def main(args):
    model = AntibodyPopulation(args=args)
    model.initialization()
    for g in range(args.G):
        model.update_dominant_population()
        model.serialize_individuals()
        if model.stop_condition(current_generation=g):
            break
        print('Generation: {0}/{1}'.format(g + 1, args.Gmax))
        model.show_bands()
        model.serialize_individuals()
        model.active_population_selection()
        model.clone_crossover_mutation(generation_idx=g)
        model.update_antibody_population()
        model.end_generation()
    print('Final {0} selected bands:'.format(args.bands_per_antibody))
    model.show_bands()
    model.serialize_individuals()


if __name__ == '__main__':
    main(arguments())
