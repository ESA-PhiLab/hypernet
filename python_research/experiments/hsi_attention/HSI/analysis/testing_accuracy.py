import glob
import pickle

import numpy as np

def get_accuracy(runs):
    epochs = []

    for run in runs:
        h = pickle.load(open(run, 'rb'))
        epochs.append(h)

    return np.array(epochs)


def main():
    histories = [history for history in
                 glob.iglob('/Users/pablo/Downloads/runners/**/*_testing_accuracy.pkl', recursive=True)
                 if 'pavia' in history and 'no_attention' in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_accuracy(modules_2)
    epochs_3 = get_accuracy(modules_3)
    epochs_4 = get_accuracy(modules_4)

    print('PAVIA - NO ATTENTION')
    print('\n- 2 modules:\n')

    epochs_2_mean = np.mean(epochs_2, axis=0)
    epochs_2_std = np.std(epochs_2, axis=0)

    for i in range(epochs_2.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_2_mean[i], epochs_2_std[i]))

    print('\n- 3 modules:\n')

    epochs_3_mean = np.mean(epochs_3, axis=0)
    epochs_3_std = np.std(epochs_3, axis=0)

    for i in range(epochs_3.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_3_mean[i], epochs_3_std[i]))

    print('\n- 4 modules:\n')

    epochs_4_mean = np.mean(epochs_4, axis=0)
    epochs_4_std = np.std(epochs_4, axis=0)

    for i in range(epochs_4.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_4_mean[i], epochs_4_std[i]))


    histories = [history for history in
                 glob.iglob('/Users/pablo/Downloads/runners/**/*_testing_accuracy.pkl', recursive=True)
                 if 'salinas' in history and 'no_attention' in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_accuracy(modules_2)
    epochs_3 = get_accuracy(modules_3)
    epochs_4 = get_accuracy(modules_4)

    print('SALINAS - NO ATTENTION')
    print('\n- 2 modules:\n')

    epochs_2_mean = np.mean(epochs_2, axis=0)
    epochs_2_std = np.std(epochs_2, axis=0)

    for i in range(epochs_2.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_2_mean[i], epochs_2_std[i]))

    print('\n- 3 modules:\n')

    epochs_3_mean = np.mean(epochs_3, axis=0)
    epochs_3_std = np.std(epochs_3, axis=0)

    for i in range(epochs_3.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_3_mean[i], epochs_3_std[i]))

    print('\n- 4 modules:\n')

    epochs_4_mean = np.mean(epochs_4, axis=0)
    epochs_4_std = np.std(epochs_4, axis=0)

    for i in range(epochs_4.shape[-1]):
        print('C{}: {} +-{}'.format(i+1, epochs_4_mean[i], epochs_4_std[i]))

if __name__ == '__main__':
    main()