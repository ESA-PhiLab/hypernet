import pickle
import os
import argparse
import numpy as np
import torch
from torch.nn.functional import relu
from sklearn.neighbors import LocalOutlierFactor


def arguments():

    parser = argparse.ArgumentParser(description='Input  arguments.')

    parser.add_argument('--dataset',
                        action="store",
                        dest="dataset",
                        type=str,
                        help='Dataset')

    parser.add_argument('--class_num',
                        action="store",
                        dest="class_num",
                        type=int,
                        help='Number of classes')

    parser.add_argument('--bands_num',
                        action="store",
                        dest="bands_num",
                        type=int,
                        help='Number of bands')

    return parser.parse_args()


def main():
    # Bands and Final are the attention bands -1 indicates anomaly and 0 what we reject.

    data_set = arguments().dataset
    dir_name = "artifacts"  # The dir name.
    RUNS = 5  # How many runs was there.
    class_num = arguments().class_num
    shape = arguments().bands_num

    bands = []

    final = []

    for i, _ in enumerate([2, 3, 4]):

        module_bands = []

        for j in range(RUNS):

            x = pickle.load(open(os.path.join(dir_name + "\\" + data_set + "_" + str(_) +
                                              "_modules_run_" + str(j + 1) + "\\" +
                                              data_set +
                                              "_attention_bands.pkl"), "rb"))

            maps = x['attention_results']
            classes = x['y_test']

            v = [[] for h in range(class_num)]

            for n, m in enumerate(classes):

                m = np.asarray(m, dtype=np.float).reshape(1, m.size)
                m = torch.from_numpy(np.asarray(m))
                m = m.type(torch.FloatTensor)

                if _ == 2:
                    v[int(m.max(1)[1].type_as(m))].append(np.average([maps[0][n],
                                                                      maps[1][n]], axis=0))
                elif _ == 3:
                    v[int(m.max(1)[1].type_as(m))].append(np.average([maps[0][n],
                                                                      maps[1][n],
                                                                      maps[2][n]],
                                                                     axis=0))
                else:
                    v[int(m.max(1)[1].type_as(m))].append(np.average([maps[0][n],
                                                                      maps[1][n],
                                                                      maps[2][n],
                                                                      maps[3][n]],
                                                                     axis=0))

            v = np.asarray([np.average(v[n], axis=0) for n in range(len(v))])

            v = torch.from_numpy(v)
            v = relu(v)
            module_bands.append(np.asarray(v))

        bands.append(np.average(module_bands, axis=0))

    clf = LocalOutlierFactor()

    for i in bands:
        Y = np.asarray([clf.fit_predict(i[n].reshape(shape, 1)) for n in range(class_num)])
        final.append(Y)

    bands = [(bands[i] > 0.05) * bands[i] for i in range(3)]
    for i in range(3):
        bands[i][bands[i] > 0.05] = -1

    return bands, final


if __name__ == "__main__":
    main()