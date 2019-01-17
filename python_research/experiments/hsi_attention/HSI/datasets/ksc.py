import numpy as np
import scipy.io
import os


def load_ksc():
    samples = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"KSC.mat"))
    ground_truth = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"KSC_gt.mat"))

    x_ksc = []
    y_ksc = []

    for i in range(samples['KSC'].shape[0]):
        for j in range(samples['KSC'].shape[1]):
            x_ksc.append(samples['KSC'][i][j].reshape((1, 176)))
            c = ground_truth['KSC_gt'][i][j]
            x = np.zeros((1, 14))
            x[0][c] = 1
            x = x.reshape((1, 14))
            y_ksc.append(x)

    return np.array(x_ksc), np.array(y_ksc)