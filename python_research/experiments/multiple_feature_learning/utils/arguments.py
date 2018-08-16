import argparse


def parse_multiple_features():
    parser = argparse.ArgumentParser(description="Script for "
                                                 "Multiple Feature Learning")
    parser.add_argument('-q',
                        action="store",
                        default="indiana",
                        dest='dataset_name',
                        type=str,
                        help="Name of the dataset for which the model "
                             "data will be loaded")
    parser.add_argument('-o',
                        action='store',
                        dest="original_path",
                        type=str,
                        help="Path to the original dataset in .npy format")
    parser.add_argument('-a',
                        action='store',
                        dest="area_path",
                        type=str,
                        help="Path to the EAP dataset for area attribute "
                             "in .npy format")
    parser.add_argument('-s',
                        action='store',
                        dest="stddev_path",
                        type=str,
                        help="Path to the EAP dataset for standard deviation "
                             "attribute in .npy format")
    parser.add_argument('-d',
                        action='store',
                        dest="diagonal_path",
                        type=str,
                        help="Path to the EAP dataset for diagonal attribute "
                             "in .npy format")
    parser.add_argument('-m',
                        action='store',
                        dest="moment_path",
                        type=str,
                        help="Path to the EAP dataset for moment attribute "
                             "in .npy format")
    parser.add_argument('-t',
                        action="store",
                        dest="gt_path",
                        type=str,
                        help="Path to the ground truth file in .npy format")
    parser.add_argument("-r",
                        action="store",
                        dest='output_dir',
                        type=str,
                        help="Path to the output directory in which artifacts "
                             "will be stored")
    parser.add_argument("-z",
                        action="store",
                        dest="output_name",
                        type=str,
                        help="Name of the output file in which data will "
                             "be stored")
    parser.add_argument('-n',
                        action="store",
                        dest="neighbourhood",
                        nargs="+",
                        type=int,
                        help="Neighbourhood size of the pixel")
    parser.add_argument('-w',
                        action='store',
                        dest='nb_samples',
                        type=int,
                        help="Number of training samples used")
    parser.add_argument('-b',
                        action="store",
                        dest="batch_size",
                        type=int,
                        help='Size of training batch')
    parser.add_argument('-p',
                        action="store",
                        dest="patience",
                        type=int,
                        help='Number of epochs without improvement on '
                             'validation score before stopping the learning')
    parser.add_argument('-v',
                        action="store",
                        dest="verbosity",
                        type=int,
                        default=1,
                        help='Verbosity of training')
    return parser.parse_args()


def parse_single_feature():
    parser = argparse.ArgumentParser(description="Script for "
                                                 "Single Feature Learning")
    parser.add_argument('-q',
                        action="store",
                        default="indiana",
                        dest='dataset_name',
                        type=str,
                        help="Name of the dataset for which the model "
                             "data will be loaded")
    parser.add_argument('-o',
                        action='store',
                        dest="data_path",
                        type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('-t',
                        action="store",
                        dest="gt_path",
                        type=str,
                        help="Path to the ground truth file in .npy format")
    parser.add_argument("-r",
                        action="store",
                        dest='output_dir',
                        type=str,
                        help="Path to the output directory in which artifacts "
                             "will be stored")
    parser.add_argument("-z",
                        action="store",
                        dest="output_name",
                        type=str,
                        help="Name of the output file in which data will "
                             "be stored")
    parser.add_argument('-n',
                        action="store",
                        dest="neighbourhood",
                        nargs="+",
                        type=int,
                        help="Neighbourhood size of the pixel")
    parser.add_argument('-w',
                        action='store',
                        dest='nb_samples',
                        type=float,
                        help="Number of training samples used")
    parser.add_argument('-b',
                        action="store",
                        dest="batch_size",
                        type=int,
                        help='Size of training batch')
    parser.add_argument('-p',
                        action="store",
                        dest="patience",
                        type=int,
                        help='Number of epochs without improvement on '
                             'validation score before stopping the learning')
    parser.add_argument('-v',
                        action="store",
                        dest="verbosity",
                        type=int,
                        default=1,
                        help='Verbosity of training')
    return parser.parse_args()


def parse_grids():
    parser = argparse.ArgumentParser(description="Script for grids")
    parser.add_argument('-q',
                        action="store",
                        default="indiana",
                        dest='dataset_name',
                        type=str,
                        help="Name of the dataset for which the model "
                             "data will be loaded")
    parser.add_argument('-d',
                        action="store",
                        dest='dir',
                        type=str,
                        help="Path to directory containing all patches along "
                             "with their respective ground truths. "
                             "This directory should also contain test set.")
    parser.add_argument("-r",
                        action="store",
                        dest='output_dir',
                        type=str,
                        help="Path to the output directory in which artifacts "
                             "will be stored")
    parser.add_argument("-z",
                        action="store",
                        dest="output_name",
                        type=str,
                        help="Name of the output file in which data will "
                             "be stored")
    parser.add_argument('-n',
                        action="store",
                        dest="neighbourhood",
                        nargs="+",
                        type=int,
                        default=[1, 1],
                        help="Neighbourhood size of the pixel")
    parser.add_argument('-w',
                        action='store',
                        dest='nb_samples',
                        type=int,
                        help="Number of training samples used")
    parser.add_argument('-b',
                        action="store",
                        dest="batch_size",
                        type=int,
                        help='Size of training batch')
    parser.add_argument('-p',
                        action="store",
                        dest="patience",
                        type=int,
                        help='Number of epochs without improvement on '
                             'validation score before stopping the learning')
    parser.add_argument('-v',
                        action="store",
                        dest="verbosity",
                        type=int,
                        default=1,
                        help='Verbosity of training')
    parser.add_argument('-t',
                        action="store",
                        dest="kernels",
                        type=int,
                        default=1,
                        help='Number of kernels in convolutional layers')
    parser.add_argument('-o',
                        action="store",
                        dest="kernel_size",
                        type=int,
                        default=1,
                        help='Size of the kernel in convolutional layers')
    return parser.parse_args()