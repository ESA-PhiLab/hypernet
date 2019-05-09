import argparse
from typing import NamedTuple

import numpy as np
import torch

from python_research.experiments.sota_models.conv_3D import conv_3D
from python_research.experiments.sota_models.utils.models_runner import run_model
from python_research.experiments.sota_models.utils.monte_carlo import prep_monte_carlo


class Arguments(NamedTuple):
    """
    Container for 3D Convolution runner arguments.
    """
    run_idx: str
    cont: str
    epochs: int
    data_path: str
    data_name: str
    neighborhood_size: int
    labels_path: str
    batch: int
    patience: int
    dest_path: str
    classes: int
    test_size: float
    val_size: float
    channels: list
    input_dim: int


def arguments() -> Arguments:
    """
    Collect arguments for running the 3D convolutional neural network.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for running 3D convolution.")
    parser.add_argument("--run_idx", dest="run_idx", help="Index of run.", type=int)
    parser.add_argument("--cont", dest="cont",
                        help="Path to file containing indexes of selected bands. (Optional argument).",
                        type=str)
    parser.add_argument("--epochs", dest="epochs", help="Total number of epochs.", type=int)
    parser.add_argument("--data_path", dest="data_path", help="Path to the dataset.", type=str)
    parser.add_argument("--data_name", dest="data_name", help="Name of the dataset.", type=str)
    parser.add_argument("--neighborhood_size", dest="neighborhood_size", help="Spatial size of the patch."
                                                                              "Due to the number of layers,"
                                                                              "the minimal size is 7.", type=int)
    parser.add_argument("--labels_path", dest="labels_path", help="Path to labels.", type=str)
    parser.add_argument("--batch", dest="batch", help="Size of the batch.", type=int)
    parser.add_argument("--patience", dest="patience", help="Number of epochs without improvement.", type=int)
    parser.add_argument("--dest_path", dest="dest_path", help="Destination to the the artifacts folder.", type=str)
    parser.add_argument("--classes", dest="classes", help="Number of classes.", type=int)
    parser.add_argument("--test_size", dest="test_size", help="Size of the test subset.", type=float)
    parser.add_argument("--val_size", dest="val_size", help="Size of the validation subset.", type=float)
    parser.add_argument("--channels", dest="channels", nargs="+",
                        help="List of channels in each convolutional layer, e.g. \"--channels 1 2 3\"",
                        required=True)
    parser.add_argument("--input_dim", dest="input_dim", nargs="+",
                        help="Dimensionality of the input sample, e.g. \"--input_dim (number_of_channels) 7 7\"",
                        required=True)
    return Arguments(**vars(parser.parse_args()))


def main(args: Arguments):
    """
    Main method used for running the 3D convolution.

    :param args: Parsed arguments.
    """
    device = torch.device("cpu")
    dtype = "torch.FloatTensor"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        dtype = "torch.cuda.FloatTensor"
    model = conv_3D.ConvNet3D(classes=args.classes, channels=list(map(int, args.channels)),
                              input_dim=np.asarray(list(map(int, args.input_dim))),
                              batch_size=args.batch, dtype=dtype).to(device=device)
    if torch.cuda.is_available():
        model = model.cuda()
    run_model(args=args, model=model, data_prep_function=prep_monte_carlo)


if __name__ == "__main__":
    parsed_args = arguments()
    main(args=parsed_args)
