import argparse
from typing import NamedTuple

import torch

from python_research.experiments.sota_models.bass.bass import Bass
from python_research.experiments.sota_models.utils.models_runner import run_model
from python_research.experiments.sota_models.utils.sets_by_sizes import prep_sets_by_sizes


class Arguments(NamedTuple):
    """
    Container for BASS runner arguments.
    """
    run_idx: str
    cont: str
    epochs: int
    data_path: str
    data_name: str
    neighborhood_size: int
    batch: int
    train_size: int
    val_size: int
    patience: int
    nb: int
    in_channels: int
    out_channels: int
    labels_path: str
    dest_path: str
    classes: int


def arguments() -> Arguments:
    """
    Collect arguments for BASS model.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Input arguments for runner.")
    parser.add_argument("--run_idx", dest="run_idx", help="Index of the run.", type=int)
    parser.add_argument("--cont", dest="cont",
                        help="Path to file containing indexes of selected bands. (Optional argument).",
                        type=str)
    parser.add_argument("--epochs", dest="epochs", help="Total number of epochs.", type=int)
    parser.add_argument("--data_path", dest="data_path", help="Path to the dataset.", type=str)
    parser.add_argument("--data_name", dest="data_name", help="Name of the data set.", type=str)
    parser.add_argument("--neighborhood_size", dest="neighborhood_size",
                        help="Spatial size of the patch. The default is 3.", type=int, default=3)
    parser.add_argument("--batch", dest="batch", help="Size of the batch for the model.", type=int)
    parser.add_argument("--train_size", dest="train_size", help="Train size as per-class number of samples.", type=int)
    parser.add_argument("--val_size", dest="val_size", help="Val size as per-class number of samples.", type=int)
    parser.add_argument("--patience", dest="patience", help="Number of epochs without improvement.", type=int,
                        default=60)
    parser.add_argument("--nb", dest="nb", type=int,
                        help="Number of convolutional blocks in the second block of the network,"
                             "i.e. 14 and 5 for \"Salinas Valley\" and \"Pavia University\" respectively.")
    parser.add_argument("--in_channels", type=int, dest="in_channels",
                        help="Number of input channels for first block of the network.")
    parser.add_argument("--out_channels", type=int, dest="out_channels",
                        help="Number of output channels for first block of the network.")
    parser.add_argument("--labels_path", dest="labels_path", help="Path to the file with labels.", type=str)
    parser.add_argument("--dest_path", dest="dest_path", help="Path to destination folder.", type=str)
    parser.add_argument("--classes", dest="classes", help="Number of classes for the model.", type=int)
    return Arguments(**vars(parser.parse_args()))


def main(args: Arguments):
    """
    Main method used for running the model.

    :param args: Parsed arguments.
    """
    device = torch.device("cpu")
    dtype = "torch.FloatTensor"
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        dtype = "torch.cuda.FloatTensor"
    model = Bass(classes=args.classes, in_channels_in_block1=args.in_channels,
                 out_channels_in_block1=args.out_channels,
                 nb=args.nb, batch_size=args.batch, dtype=dtype,
                 neighborhood_size=args.neighborhood_size).to(device=device)
    run_model(args=args, model=model, data_prep_function=prep_sets_by_sizes)


if __name__ == "__main__":
    parsed_args = arguments()
    main(args=parsed_args)
