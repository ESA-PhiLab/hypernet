import glob
import pickle

import numpy as np

def get_average_map(maps):

    averaged_heatmaps = []
    for path in maps:
        averaged_heatmaps.append(np.mean(np.array(pickle.load(open(path, 'rb'))), axis=1).reshape((16, 224)))

    return np.mean(np.array(averaged_heatmaps), axis=0)


def main():
    heatmaps = [heatmap for heatmap in glob.iglob('C:\\Users\\pribalta.FP\\Downloads\\runners\\**\\*_attention_bands.pkl', recursive=True)
                if 'salinas' in heatmap and 'no_attention' in heatmap]


    modules_2 = [heatmap for heatmap in heatmaps if '2_modules' in heatmap]
    modules_3 = [heatmap for heatmap in heatmaps if '3_modules' in heatmap]
    modules_4 = [heatmap for heatmap in heatmaps if '4_modules' in heatmap]

    avg_2_modules = get_average_map(modules_2)
    avg_3_modules = get_average_map(modules_3)
    avg_4_modules = get_average_map(modules_4)

    import matplotlib.pyplot as plt

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.figure(figsize=(14,2))
    for i in range(avg_2_modules.shape[0]):
        ax = plt.subplot(4,4,1+i)

        plt.ylim([0.005,0.02])
        plt.xlim([0,224])

        plt.xticks([])
        plt.yticks([])

        plt.axvline(x=25, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=50, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=75, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=100, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=125, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=150, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=175, color='lightgrey', linestyle='--', linewidth=1)
        plt.axvline(x=200, color='lightgrey', linestyle='--', linewidth=1)

        if i == 12:
            plt.xlabel('Hyperspectral bands', x=2.0, fontsize=14)
            plt.ylabel('Attention score', y=2.0, fontsize=14)

        if i < 9:
            ax.text(207, 0.015, 'C' + str(i+1), size=12)
        else:
            ax.text(200,0.015,'C'+str(i+1), size=12)

        plt.plot(np.arange(avg_2_modules.shape[-1]), avg_2_modules[i]/2, '#f1a340', label='2 attention modules')
        plt.plot(np.arange(avg_2_modules.shape[-1]), avg_3_modules[i]/3, '#998ec3', label='3 attention modules')
        plt.plot(np.arange(avg_2_modules.shape[-1]), avg_4_modules[i]/4, 'r', label='4 attention modules')

    plt.legend(ncol=3,bbox_to_anchor=(0.45, -0.55), loc=1, edgecolor='k', fontsize=14)

    plt.suptitle('Attention scores for all classes across hyperspectral bands in the Salinas Valley dataset', y=1.025, fontsize=16)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('salinas.eps', bbox_inches='tight')

if __name__ == '__main__':
    main()