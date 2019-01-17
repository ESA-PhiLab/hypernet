import numpy as np
import scipy.io
import os


def load_indian_pines():
    samples = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"Indian_pines.mat"))
    ground_truth = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"Indian_pines_gt.mat"))

    x_indian = []
    y_indian = []

    for i in range(samples['indian_pines'].shape[0]):
        for j in range(samples['indian_pines'].shape[1]):
            x_indian.append(samples['indian_pines'][i][j].reshape((1, 220)))
            c = ground_truth['indian_pines_gt'][i][j]
            x = np.zeros((1, 17))
            x[0][c] = 1
            x = x.reshape((1, 17))
            y_indian.append(x)

    return np.array(x_indian), np.array(y_indian)