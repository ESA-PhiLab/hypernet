import matplotlib.pyplot as plt
import numpy as np


def plot_heatmaps(heatmaps: np.ndarray):
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(heatmaps)
    axis.set_yticklabels([str(class_ + 1) for class_ in range(heatmaps.shape[0])], minor=False)
    plt.colorbar(heatmap)
    fig.set_size_inches(10, 5)
    plt.title('Attention heatmaps scores')
    plt.ylabel('Class indexes')
    plt.xlabel('Band spectrum')
    plt.show()
