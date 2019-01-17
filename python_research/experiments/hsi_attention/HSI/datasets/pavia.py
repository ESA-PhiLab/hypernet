import os
import numpy as np
import scipy.io


def load_pavia_university():
    samples = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"PaviaU.mat"))
    ground_truth = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"PaviaU_gt.mat"))

    x_pavia = []
    y_pavia = []
    for i in range(samples['paviaU'].shape[0]):
        for j in range(samples['paviaU'].shape[1]):
            x_pavia.append(samples['paviaU'][i][j].reshape((1, 103)))
            c = ground_truth['paviaU_gt'][i][j]
            x = np.zeros((1, 10))
            x[0][c] = 1
            x = x.reshape((1, 10))
            y_pavia.append(x)

    return np.array(x_pavia), np.array(y_pavia)