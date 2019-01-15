import argparse

import numpy as np
import torch

from python_research.experiments.sota_models.conv_3D import conv_3D
from python_research.experiments.sota_models.utils.models_runner import run_model
from python_research.experiments.sota_models.utils.monte_carlo import prep_monte_carlo


def arguments():
    """
    Arguments for running the experiments on 3D convolutional neural network.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for runner.')
    parser.add_argument('--run_idx', dest='run_idx', help='Run index.')
    parser.add_argument('--dtype', dest='dtype', help='Data type used by the model.')
    parser.add_argument('--cont', dest='cont', help='Path to file containing indexes of selected bands.', type=str)
    parser.add_argument('--epochs', dest='epochs', help='Number of epochs.', type=int)
    parser.add_argument('--data_path', dest='data_path', help='Path to the data set.')
    parser.add_argument('--data_name', dest='data_name', help='Name of the data set.')
    parser.add_argument('--neighborhood_size', dest='neighborhood_size', help='Spatial size of the patch.', type=int)
    parser.add_argument('--labels_path', dest='labels_path', help='Path to labels.')
    parser.add_argument('--batch', dest='batch', help='Batch size.', type=int)
    parser.add_argument('--patience', dest='patience', help='Number of epochs without improvement.')
    parser.add_argument('--dest_path', dest='dest_path', help='Destination to the the artifacts folder.')
    parser.add_argument('--classes', dest='classes', help='Number of classes.', type=int)
    parser.add_argument('--test_size', dest='test_size', help='Test size.', type=float)
    parser.add_argument('--val_size', dest='val_size', help='Validation size.', type=float)
    parser.add_argument('--channels', dest='channels', nargs='+',
                        help='List of channels. Format: --channels 1 2 3',
                        required=True)
    parser.add_argument('--input_dim', dest='input_dim', nargs='+',
                        help='Input dimensionality. Format: --channels spectral_dim_size 7 7',
                        required=True)
    return parser.parse_args()


def main(args):
    """
    Create model.

    :param args: Parsed arguments.
    """
    model = conv_3D.ConvNet3D(classes=args.classes, channels=list(map(int, args.channels)),
                              input_dim=np.asarray(list(map(int, args.input_dim))),
                              batch_size=args.batch, dtype=args.dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    run_model(args=args, model=model, data_prep_function=prep_monte_carlo)


if __name__ == '__main__':
    main(arguments())
