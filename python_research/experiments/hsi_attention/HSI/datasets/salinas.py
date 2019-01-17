import numpy as np
import scipy.io
import os


def load_salinas():
    samples = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"salinas.mat"))
    ground_truth = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), r"Salinas_gt.mat"))

    x_salinas = []
    y_salinas = []

    for i in range(samples['salinas'].shape[0]):
        for j in range(samples['salinas'].shape[1]):
            x_salinas.append(samples['salinas'][i][j].reshape((1,224)))
            c = ground_truth['salinas_gt'][i][j]
            x = np.zeros((1, 17))
            x[0][c] = 1
            x = x.reshape((1, 17))
            y_salinas.append(x)

    return np.array(x_salinas), np.array(y_salinas)