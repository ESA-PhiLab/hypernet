from python_research.experiments.band_selection_algorithms.BOMBS.immune_system_based_model import AntibodyPopulation
from python_research.experiments.band_selection_algorithms.BS_IC.utils import *


def arguments():
    """
    Collect all arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for data loader.')
    parser.add_argument('--G', dest='G', type=int, help='Number of generations.', default=99999999)
    parser.add_argument('--Gmax', dest='Gmax', type=int, help='Max. number of generations.', default=100)
    parser.add_argument('--Na', dest='Na', type=int, help='Max. size of active population.', default=20)
    parser.add_argument('--Nd', dest='Nd', type=int, help='Max. size of dominant population.', default=100)
    parser.add_argument('--Nc', dest='Nc', type=int, help='Max. size of clone population.', default=100)
    parser.add_argument('--TD_size', dest='TD_size', type=int, help='Initial size of dominant population.', default=110)
    parser.add_argument('--P_init_size', dest='P_init_size', type=int,
                        help='Initial size of population P.', default=200)
    parser.add_argument('--bands_per_antibody', dest='bands_per_antibody', type=int, help='Number of bands per antibody'
                                                                                          'PaviaU: 20'
                                                                                          'Salinas: 21')
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--ref_map_path', dest='ref_map_path', type=str)
    parser.add_argument('--dest_path', dest='dest_path', type=str, help='Destination path for selected bands.')
    return parser.parse_args()


def run(args):
    """
    Main loop of the band selection algorithm.
    """
    model = AntibodyPopulation(args=args)
    model.initialization()
    for g in range(args.G):
        model.update_dominant_population()
        model.serialize_individuals()
        if model.stop_condition(current_generation=g):
            break
        print('Generation: {0}/{1}'.format(g + INCREMENT_ONE, args.Gmax))
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
    run(arguments())
