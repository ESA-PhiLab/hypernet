import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from python_research.experiments.band_selection_algorithms.MI.spectral_band import *
from python_research.experiments.band_selection_algorithms.utils import *
from python_research.experiments.band_selection_algorithms.MI.mutual_information import *


def arg_parser():
    """
    Arguments for running the mutual information based band selection algorithm.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for mutual information band selection.")
    parser.add_argument("--data_path", dest="data_path", type=str)
    parser.add_argument("--ref_map_path", dest="ref_map_path", type=str)
    parser.add_argument("--dest_path", dest="dest_path", type=str, help="Destination path for selected bands.")
    parser.add_argument("--X", dest="X", type=int, default=20)
    parser.add_argument("--b", dest="b", type=int, default=3,
                        help="This parameter referred in the paper as: \"rejection bandwidth\" is dataset dependent."
                             "For Pavia University can be 3 and for Salinas Valley 5.")
    parser.add_argument("--eta", dest="eta", type=float, default=0.007,
                        help="This parameter referred in the paper as: \"complementary threshold\" is dataset dependent.")
    return parser.parse_args()


def main(args):
    """
    Contains all steps of the band selection algorithm.

    :param args: Parsed arguments.
    """
    data, ref_map = load_data(data_path=args.data_path, ref_map_path=args.ref_map_path)
    mutual_info_band_selector = MutualInformation(designed_band_size=args.X, b=args.b, eta=args.eta)
    mutual_info_band_selector.prep_bands(data=data, ref_map=ref_map)
    mutual_info_band_selector.calculate_mi(dest_path=args.dest_path)
    mutual_info_band_selector.perform_search()
    np.savetxt(fname=os.path.join(args.dest_path, "chosen_bands"),
               X=np.sort(np.asarray(mutual_info_band_selector.set_of_selected_bands)), fmt="%d")
    print("Selected bands: {}".format(np.sort(np.asarray(mutual_info_band_selector.set_of_selected_bands))))


if __name__ == "__main__":
    main(arg_parser())
