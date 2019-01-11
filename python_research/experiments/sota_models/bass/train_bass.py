import argparse

import torch

from python_research.experiments.sota_models.bass.bass import Bass
from python_research.experiments.sota_models.utils.models_runner import run_model
from python_research.experiments.sota_models.utils.sets_by_sizes import prep_sets_by_sizes


def arguments():
    """
    Arguments for BASS model.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Input arguments for runner.')
    parser.add_argument('--dtype', dest='dtype', help='Data type used by the model.')
    parser.add_argument('--cont', dest='cont', help='Path to file containing indexes of selected bands.', type=str)
    parser.add_argument('--run_idx', dest='run_idx', help='Run index.')
    parser.add_argument('--epochs', dest='epochs', help='Number of epochs.', type=int)
    parser.add_argument('--data_path', dest='data_path', help='Path to the data set.')
    parser.add_argument('--data_name', dest='data_name', help='Name of the data set.')
    parser.add_argument('--neighbourhood_size', dest='neighbourhood_size',
                        help='Spatial size of the patch. The default is 3.', type=int, default=3)
    parser.add_argument('--batch', dest='batch', help='Batch size.', type=int)
    parser.add_argument('--train_size', dest='train_size', help='Train size.', type=int)
    parser.add_argument('--val_size', dest='val_size', help='Val size.', type=int)
    parser.add_argument('--patience', dest='patience', help='Number of epochs without improvement.')
    parser.add_argument('--nb', dest='nb', type=int,
                        help='Number of chunks in block 2: 14 and 5 for Salinas and PaviaU respectively.')
    parser.add_argument('--in_channels', type=int, dest='in_channels', help='Number of input channels for first block.')
    parser.add_argument('--out_channels', type=int, dest='out_channels',
                        help='Number of output channels for first block.')
    parser.add_argument('--labels_path', dest='labels_path', help='Path to labels.')
    parser.add_argument('--dest_path', dest='dest_path', help='Destination of the artifacts folder.')
    parser.add_argument('--classes', dest='classes', help='Number of classes.', type=int)
    return parser.parse_args()


def main(args):
    """
    Create model.

    :param args: Parsed arguments.
    """
    model = Bass(classes=args.classes, in_channels_in_block1=args.in_channels,
                 out_channels_in_block1=args.out_channels,
                 nb=args.nb, batch_size=args.batch, dtype=args.dtype, neighbourhood_size=args.neighbourhood_size)
    if torch.cuda.is_available():
        model = model.cuda()
    run_model(args=args, model=model, data_prep_function=prep_sets_by_sizes)


if __name__ == '__main__':
    main(arguments())
