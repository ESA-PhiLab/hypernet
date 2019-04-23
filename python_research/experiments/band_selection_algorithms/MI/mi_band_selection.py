import argparse
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from python_research.experiments.band_selection_algorithms.MI.spectral_band import SpectralBand
from python_research.experiments.band_selection_algorithms.utils import *


class MutualInformation(object):
    def __init__(self, designed_band_size: int, b: int, eta: float):
        """
        Initialize all instance variables.

        :param designed_band_size: Number of bands to select.
        :param b: The neighborhood of selected band - bandwidth.
        :param eta: Threshold which prevents from redundancy in the selected bands.
        """
        self.ref_map = None
        self.ref_map_hist = None
        self.set_of_selected_bands = []
        self.designed_band_size = designed_band_size
        self.set_of_remaining_bands = []
        self.b = b
        self.eta = eta
        self.band_size = None

    def return_mi_scores(self):
        return [obj.mutual_information for obj in self.set_of_remaining_bands]

    def select_band_index(self):
        """
        Select band indexes by choosing the argmax from the mutual information collection.
        """
        selected_band = self.set_of_remaining_bands[np.argmax(self.return_mi_scores()).astype(int)]
        neighbor_set = list(range(int(self.set_of_remaining_bands.index(selected_band) - (self.b + 1)),
                                  int(self.set_of_remaining_bands.index(selected_band) + self.b + 1)))
        if neighbor_set[0] < 0:
            neighbor_set[0] = 0
        if any(elem > self.set_of_remaining_bands.__len__() for elem in neighbor_set):
            neighbor_set = neighbor_set[:neighbor_set.index(self.set_of_remaining_bands.__len__())]
        delta_mi = []
        for i in neighbor_set[:-1]:
            delta_mi.append(abs(self.set_of_remaining_bands[i + 1].mutual_information -
                                self.set_of_remaining_bands[i].mutual_information))
        if max(delta_mi) < self.eta:
            for band_to_be_deleted in sorted(neighbor_set, reverse=True):
                self.set_of_remaining_bands.pop(band_to_be_deleted)
        else:
            self.set_of_remaining_bands.pop(self.set_of_remaining_bands.index(selected_band))
        self.set_of_selected_bands.append(selected_band.band_index)
        assert self.set_of_remaining_bands.__len__() >= \
               neighbor_set.__len__(), "Error, either \"rejection bandwidth\"" \
                                       " parameter or \"complementary threshold\"" \
                                       " was set to high," \
                                       " those parameters are dataset dependent.\n" \
                                       "Please, check those parameters and set them correctly."

    def perform_search(self):
        """
        Main loop for mutual information - based band selection algorithm.
        """
        while self.set_of_selected_bands.__len__() < self.designed_band_size:
            self.select_band_index()
        assert np.unique(
            self.set_of_selected_bands).__len__() == self.set_of_selected_bands.__len__(), \
            "The \"b - Beta\" parameter was set to high together with the number of bands to be selected."
        self.set_of_selected_bands = np.sort(self.set_of_selected_bands)

    def calculate_mi(self, dest_path: str = None):
        """
        Calculate mutual information between the reference image and each band in the hyperspectral data block.

        :param dest_path: Destination path for MI plot.
        """
        h_b = -np.sum(np.dot(self.ref_map_hist, np.ma.log2(self.ref_map_hist)))
        for i in range(len(self.set_of_remaining_bands)):
            h_a = -np.sum(np.dot(self.set_of_remaining_bands[i].histogram,
                                 np.ma.log2(self.set_of_remaining_bands[i].histogram)))
            h_ab = -np.sum(np.multiply(self.set_of_remaining_bands[i].joint_histogram,
                                       np.ma.log2(self.set_of_remaining_bands[i].joint_histogram)))
            self.set_of_remaining_bands[i].set_mutual_information(h_a + h_b - h_ab)

        if dest_path is not None:
            plt.plot(self.return_mi_scores())
            plt.title("Mutual information of each band according to the reference map")
            plt.ylabel("MI based on reference map")
            plt.xlabel("Spectral bands")
            plt.savefig(os.path.join(dest_path, 'mi_plot'))

    def prep_bands(self, data: np.ndarray, ref_map: np.ndarray):
        """
        Prepare bands, grey level histograms and joint histograms.

        :param data: Data block.
        :param ref_map: Reference map.
        """
        self.band_size = data.shape[SPECTRAL_AXIS] - 1
        non_zeros = np.nonzero(ref_map)
        self.ref_map = ref_map[non_zeros]
        self.ref_map_hist = np.histogram(self.ref_map.flatten(),
                                         (self.ref_map.max() - 1))[0] / self.ref_map.size
        min_max_scaler = MinMaxScaler(feature_range=(0, 255))
        for i in range(data.shape[SPECTRAL_AXIS]):
            band = np.asarray(min_max_scaler.fit_transform(data[..., i])).astype(int)
            band = band[non_zeros]
            histogram = np.histogram(band, 256)[0] / band.size
            joint_histogram = np.histogram2d(x=band,
                                             y=self.ref_map,
                                             bins=[256, (self.ref_map.max() + BG_CLASS)])[0] / self.ref_map.size
            self.set_of_remaining_bands.append(SpectralBand(histogram=histogram,
                                                            joint_histogram=joint_histogram,
                                                            band_index=i))


def arg_parser():
    """
    Arguments for running the mutual information based band selection algorithm.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Arguments for mutual information band selection.")
    parser.add_argument("--data_path", dest="data_path", type=str, help="Path to data.")
    parser.add_argument("--ref_map_path", dest="ref_map_path", type=str, help="Path to ground truth.")
    parser.add_argument("--dest_path", dest="dest_path", type=str, help="Destination path for selected bands.")
    parser.add_argument("--X", dest="X", type=int, help="Number of bands to select.")
    parser.add_argument("--b", dest="b", type=int, default=3,
                        help="This parameter referred in the paper as: \"rejection bandwidth\" is dataset dependent."
                             "For Pavia University can be 3 and for Salinas Valley 5.")
    parser.add_argument("--eta", dest="eta", type=float, default=0.007,
                        help="This parameter referred in the paper as: \"complementary threshold\" "
                             "is dataset dependent.")
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
