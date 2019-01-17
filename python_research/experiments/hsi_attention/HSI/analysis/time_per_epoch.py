import glob
import pickle

import numpy as np


def get_time_per_epoch(runs):
    epochs = []

    for run in runs:
        h = pickle.load(open(run, 'rb'))
        epochs.extend(h)

    return np.array(epochs)

def main():
    time_per_epoch = [history for history in glob.iglob('/Users/pablo/Downloads/runners/**/*_time_training.pkl', recursive=True)
                if 'pavia' in history and 'no_attention' in history]

    modules_2 = [h for h in time_per_epoch if '2_modules' in h]
    modules_3 = [h for h in time_per_epoch if '3_modules' in h]
    modules_4 = [h for h in time_per_epoch if '4_modules' in h]

    epochs_2 = get_time_per_epoch(modules_2)
    epochs_3 = get_time_per_epoch(modules_3)
    epochs_4 = get_time_per_epoch(modules_4)

    print('PAVIA - NO ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_2)))
    print('\tstd: {} seconds'.format(np.std(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_3)))
    print('\tstd: {} seconds'.format(np.std(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_4)))
    print('\tstd: {} seconds'.format(np.std(epochs_4)))

    time_per_epoch = [history for history in glob.iglob('/Users/pablo/Downloads/runners/**/*_time_training.pkl', recursive=True)
                if 'pavia' in history and 'no_attention' not in history]

    modules_2 = [h for h in time_per_epoch if '2_modules' in h]
    modules_3 = [h for h in time_per_epoch if '3_modules' in h]
    modules_4 = [h for h in time_per_epoch if '4_modules' in h]

    epochs_2 = get_time_per_epoch(modules_2)
    epochs_3 = get_time_per_epoch(modules_3)
    epochs_4 = get_time_per_epoch(modules_4)

    print()
    print('PAVIA - WITH ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_2)))
    print('\tstd: {} seconds'.format(np.std(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_3)))
    print('\tstd: {} seconds'.format(np.std(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_4)))
    print('\tstd: {} seconds'.format(np.std(epochs_4)))


    time_per_epoch = [history for history in
                      glob.iglob('/Users/pablo/Downloads/runners/**/*_time_training.pkl', recursive=True)
                      if 'salinas' in history and 'no_attention' in history]

    modules_2 = [h for h in time_per_epoch if '2_modules' in h]
    modules_3 = [h for h in time_per_epoch if '3_modules' in h]
    modules_4 = [h for h in time_per_epoch if '4_modules' in h]

    epochs_2 = get_time_per_epoch(modules_2)
    epochs_3 = get_time_per_epoch(modules_3)
    epochs_4 = get_time_per_epoch(modules_4)

    print('SALINAS - NO ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_2)))
    print('\tstd: {} seconds'.format(np.std(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_3)))
    print('\tstd: {} seconds'.format(np.std(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_4)))
    print('\tstd: {} seconds'.format(np.std(epochs_4)))

    time_per_epoch = [history for history in
                      glob.iglob('/Users/pablo/Downloads/runners/**/*_time_training.pkl', recursive=True)
                      if 'salinas' in history and 'no_attention' not in history]

    modules_2 = [h for h in time_per_epoch if '2_modules' in h]
    modules_3 = [h for h in time_per_epoch if '3_modules' in h]
    modules_4 = [h for h in time_per_epoch if '4_modules' in h]

    epochs_2 = get_time_per_epoch(modules_2)
    epochs_3 = get_time_per_epoch(modules_3)
    epochs_4 = get_time_per_epoch(modules_4)

    print()
    print('SALINAS - WITH ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_2)))
    print('\tstd: {} seconds'.format(np.std(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_3)))
    print('\tstd: {} seconds'.format(np.std(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} seconds'.format(np.mean(epochs_4)))
    print('\tstd: {} seconds'.format(np.std(epochs_4)))
    
if __name__ == '__main__':
    main()