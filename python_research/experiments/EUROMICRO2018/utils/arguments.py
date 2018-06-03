'''
Argument parsing module
'''
import argparse


def parse():
    """Parse arguments

    """
    parser = argparse.ArgumentParser(description='Script for PSO experimentation.')

    parser.add_argument('-o',
                        action="store",
                        dest="output_dir",
                        default='out',
                        help='File path to history file for initializing the archive')

    parser.add_argument('-e',
                        action="store",
                        dest="nb_epoch",
                        type=int,
                        help='Number of training epochs')

    parser.add_argument('-s',
                        action="store",
                        dest="nb_samples",
                        default=0,
                        type=int,
                        help='Number of training samples used')

    parser.add_argument('-b',
                        action="store",
                        dest="batch_size",
                        type=int,
                        help='Size of training batch')

    parser.add_argument('-d',
                        action="store",
                        dest="dataset_name",
                        type=str,
                        default='',
                        help='Dataset name')

    parser.add_argument('-v',
                        action="store",
                        dest="verbosity",
                        type=int,
                        default=1,
                        help='Verbosity of training')

    parser.add_argument('-l',
                        action="store_true",
                        dest="local",
                        help='Run locally?')

    parser.add_argument('-r',
                        action="store_false",
                        dest="local",
                        help='Run locally?')

    parser.add_argument('-c',
                        action="store_true",
                        dest="convert",
                        help='Convert via TensorRT?')

    parser.add_argument('--no-c',
                        action="store_false",
                        dest="convert",
                        help='Convert via TensorRT?')

    return parser.parse_args()


def print_config(config: dict):
    print("Dataset loaded:", config['dataset'].capitalize())
    print("Model will be converted:", config['convert'])

    if config['nb_samples'] != 0:
        print('Using', config['nb_samples'], 'samples per class.')
    else:
        print('Using max samples per class.')
