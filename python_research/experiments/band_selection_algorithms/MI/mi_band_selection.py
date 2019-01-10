import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from python_research.experiments.band_selection_algorithms.BS_IC.utils import *

USED_BAND_FLAG = -1


class MI(object):
    def __init__(self, x: int, b: int, eta: float):
        """
        Initialize all arguments.

        :param x: Number of bands to select.
        :param b: The neighbourhood of selected band.
        :param eta: Prevents from redundancy in the selected bands.
        """
        self.ref_map = None
        self.ref_map_hist = None
        self.S = []
        self.bands_ids = []
        self.bands_histograms = []
        self.joint_entropy = []
        self.mutual_information = []
        self.x = x
        self.b = b
        self.eta = eta
        self.init_bands_size = None

    def select_band_index(self):
        """
        Select band indexes taking the maximum argument from the mutual information list.
        """
        s = np.argmax(self.mutual_information).astype(int)
        n = list(range(int(s - (self.b + 1)), int(s + self.b + 1)))
        if n[0] < 0:
            n[0] = 0
        if self.init_bands_size in n:
            n = n[:n.index(self.init_bands_size) + 1]
        d = []
        for i in n[:-1]:
            if self.mutual_information[i + 1] == USED_BAND_FLAG or self.mutual_information[i] == USED_BAND_FLAG:
                d.append(0)
            else:
                d.append(self.mutual_information[i + 1] - self.mutual_information[i])
        if max(d) < self.eta:
            for n_index in n:
                self.mutual_information[n_index] = USED_BAND_FLAG
        else:
            self.mutual_information[s] = USED_BAND_FLAG
        self.S.append(self.bands_ids[s])
        self.bands_ids[s] = USED_BAND_FLAG

    def perform_search(self):
        """
        Main loop for selecting bands.
        """
        while self.S.__len__() < self.x:
            self.select_band_index()
        assert np.unique(
            self.S).__len__() == self.S.__len__(), \
            'The "b" parameter was set to high together with the number of bands to be selected.'
        self.S = np.sort(self.S)

    def calculate_mi(self, draw_plot=False):
        h_b = -np.sum(np.dot(self.ref_map_hist, np.ma.log2(self.ref_map_hist)))
        for i in range(len(self.bands_histograms)):
            h_a = -np.sum(np.dot(self.bands_histograms[i], np.ma.log2(self.bands_histograms[i])))
            h_ab = -np.sum(np.multiply(self.joint_entropy[i], np.ma.log2(self.joint_entropy[i])))
            self.mutual_information.append(h_a + h_b - h_ab)
        if draw_plot:
            plt.plot(self.mutual_information)
            plt.title('Mutual information of each band according to the reference map.')
            plt.ylabel('MI based on reference map')
            plt.xlabel('Spectral bands')
            plt.show()

    def prep_bands(self, data: np.ndarray, ref_map: np.ndarray):
        """
        Prepare bands, grey level histograms and joint histograms.

        :param data: Data block.
        :param ref_map: Reference map.
        """
        self.init_bands_size = data.shape[CONST_SPECTRAL_AXIS] + CONST_BG_CLASS
        non_zeros = np.nonzero(ref_map)
        self.ref_map = ref_map[non_zeros]
        self.ref_map_hist = np.histogram(self.ref_map.flatten(),
                                         (self.ref_map.max() - 1))[0] / self.ref_map.size
        min_max_scaler = MinMaxScaler(feature_range=(0, 255))
        self.bands_ids = list(range(data.shape[CONST_SPECTRAL_AXIS]))
        for i in range(data.shape[CONST_SPECTRAL_AXIS]):
            band = np.asarray(min_max_scaler.fit_transform(data[..., i])).astype(int)
            band = band[non_zeros]
            self.bands_histograms.append(np.histogram(band, 256)[0] / band.size)
            self.joint_entropy.append(np.histogram2d(x=band,
                                                     y=self.ref_map,
                                                     bins=[256, (self.ref_map.max() + CONST_BG_CLASS)])
                                      [0] / self.ref_map.size)


def arg_parser():
    """
    Arguments for the band selection algorithm.
    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Arguments for mutual information band selection.')
    parser.add_argument('--data_path', dest='data_path', type=str)
    parser.add_argument('--ref_map_path', dest='ref_map_path', type=str)
    parser.add_argument('--dest_path', dest='dest_path', type=str, help='Destination path for selected bands.')
    parser.add_argument('--X', dest='X', type=int, default=20)
    parser.add_argument('--b', dest='b', type=int, default=3,
                        help='This value is data set dependent. For PaviaU can be 3 and for Salinas 5')
    parser.add_argument('--eta', dest='eta', type=float, default=0.5, help='Default value according to paper..')
    return parser.parse_args()


def load_data(path, ref_map_path, get_ref_map=True):
    """
    Load data and labels.
    Normalize data.

    :param path: Path to data.
    :param ref_map_path: Path to labels.
    :param get_ref_map: True if returning labels.
    :return: Normalized data.
    """
    data = None
    ref_map = None
    if path.endswith(".npy"):
        data = np.load(path)
    elif path.endswith(".mat"):
        mat = loadmat(path)
        for key in mat.keys():
            if "__" not in key:
                data = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    if ref_map_path.endswith(".npy"):
        ref_map = np.load(ref_map_path)
    elif ref_map_path.endswith(".mat"):
        mat = loadmat(ref_map_path)
        for key in mat.keys():
            if "__" not in key:
                ref_map = mat[key]
                break
    else:
        raise ValueError("This file type is not supported.")
    assert data is not None and ref_map_path is not None, 'There is no data to be loaded.'
    min_ = np.amin(data)
    max_ = np.amax(data)
    data = (data - min_) / (max_ - min_)
    if get_ref_map is False:
        return data
    ref_map = ref_map.astype(int) + CONST_BG_CLASS
    return data.astype(float), ref_map.astype(int)


def run(args):
    """
    Contains all steps of the band selection algorithm.

    :param args: Parsed arguments.
    """
    data, ref_map = load_data(path=args.data_path, ref_map_path=args.ref_map_path)
    mutual_info_band_selector = MI(x=args.X, b=args.b, eta=args.eta)
    mutual_info_band_selector.prep_bands(data=data, ref_map=ref_map)
    mutual_info_band_selector.calculate_mi(draw_plot=True)
    mutual_info_band_selector.perform_search()
    np.savetxt(fname=os.path.join(args.dest_path, 'chosen_bands'),
               X=np.sort(np.asarray(mutual_info_band_selector.S)), fmt='%d')
    print('Selected bands: {}'.format(np.sort(np.asarray(mutual_info_band_selector.S))))


if __name__ == '__main__':
    run(arg_parser())
