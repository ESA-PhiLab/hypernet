import argparse


def arguments() -> argparse.Namespace:
    """
    Argument parser method.

    :return: Namespace object holding attributes.
    """
    parser = argparse.ArgumentParser(description="Input  arguments.")
    parser.add_argument("--dataset_path",
                        dest="dataset_path",
                        type=str,
                        help="Path to the dataset.")
    parser.add_argument("--labels_path",
                        dest="labels_path",
                        type=str,
                        help="Path to labels.")
    parser.add_argument("--selected_bands",
                        dest="selected_bands",
                        type=str,
                        help="Path to the file containing selected bands.")
    parser.add_argument("--validation",
                        dest="validation",
                        type=float,
                        help="Proportion of validation samples from each class."
                             "Default is: 0.1 * lowest_class_population.",
                        default=0.1)
    parser.add_argument("--test",
                        action="store",
                        dest="test",
                        type=float,
                        help="Proportion of test samples from each class."
                             "Default is: 0.1 * lowest_class_population.",
                        default=0.1)
    parser.add_argument("--epochs",
                        dest="epochs",
                        type=int,
                        help="Number of epochs.",
                        default=999999)
    parser.add_argument("--modules",
                        dest="modules",
                        type=int,
                        help="Number of attention modules used by the model.")
    parser.add_argument("--patience",
                        dest="patience",
                        type=int,
                        help="Patience stopping condition.",
                        default=6)
    parser.add_argument("--output_dir",
                        dest="output_dir",
                        type=str,
                        help="Output directory.")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        type=int,
                        help="Number of samples per batch.",
                        default=200)
    parser.add_argument("--attn",
                        dest="attn",
                        type=str,
                        help="Set ""true"" when using attention, ""false"" otherwise.")
    parser.add_argument("--run_idx",
                        dest="run_idx",
                        type=str,
                        help="Index of the run.")
    parser.add_argument("--cont",
                        dest="cont",
                        type=float,
                        help="Contamination parameter for outlier detector.")
    return parser.parse_args()
