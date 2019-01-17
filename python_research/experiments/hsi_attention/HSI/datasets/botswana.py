import os
import numpy as np
import scipy.io


def load_botswana():
    samples = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"Botswana.mat"))
    ground_truth = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"Botswana_gt.mat"))

    x_botswana = []
    y_botswana = []
    for i in range(samples['Botswana'].shape[0]):
        for j in range(samples['Botswana'].shape[1]):
            x_botswana.append(samples['Botswana'][i][j].reshape((1, 145)))
            c = ground_truth['Botswana_gt'][i][j]
            x = np.zeros((1, 15))
            x[0][c] = 1
            x = x.reshape((1, 15))
            y_botswana.append(x)

    return np.array(x_botswana), np.array(y_botswana)