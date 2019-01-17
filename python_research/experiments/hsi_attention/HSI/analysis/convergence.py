import glob
import pickle
import numpy as np


def get_convergence(runs):
    epochs = []

    for run in runs:
        h = pickle.load(open(run, 'rb'))
        epochs.append(len(h))

    return np.array(epochs)

def main():
    histories = [history for history in glob.iglob('/Users/pablo/Downloads/runners/**/*_training_history.pkl', recursive=True)
                if 'pavia' in history and 'no_attention' in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_convergence(modules_2)
    epochs_3 = get_convergence(modules_3)
    epochs_4 = get_convergence(modules_4)

    print('PAVIA - NO ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_2)))
    print('\tstd: {} epochs'.format(np.std(epochs_2)))
    print('\tmax: {} epochs'.format(np.max(epochs_2)))
    print('\tmin: {} epochs'.format(np.min(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_3)))
    print('\tstd: {} epochs'.format(np.std(epochs_3)))
    print('\tmax: {} epochs'.format(np.max(epochs_3)))
    print('\tmin: {} epochs'.format(np.min(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_4)))
    print('\tstd: {} epochs'.format(np.std(epochs_4)))
    print('\tmax: {} epochs'.format(np.max(epochs_4)))
    print('\tmin: {} epochs'.format(np.min(epochs_4)))

    histories = [history for history in
                 glob.iglob('/Users/pablo/Downloads/runners/**/*_training_history.pkl', recursive=True)
                 if 'pavia' in history and 'no_attention' not in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_convergence(modules_2)
    epochs_3 = get_convergence(modules_3)
    epochs_4 = get_convergence(modules_4)

    print()
    print('PAVIA - WITH ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_2)))
    print('\tstd: {} epochs'.format(np.std(epochs_2)))
    print('\tmax: {} epochs'.format(np.max(epochs_2)))
    print('\tmin: {} epochs'.format(np.min(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_3)))
    print('\tstd: {} epochs'.format(np.std(epochs_3)))
    print('\tmax: {} epochs'.format(np.max(epochs_3)))
    print('\tmin: {} epochs'.format(np.min(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_4)))
    print('\tstd: {} epochs'.format(np.std(epochs_4)))
    print('\tmax: {} epochs'.format(np.max(epochs_4)))
    print('\tmin: {} epochs'.format(np.min(epochs_4)))

    histories = [history for history in
                 glob.iglob('/Users/pablo/Downloads/runners/**/*_training_history.pkl', recursive=True)
                 if 'salinas' in history and 'no_attention' in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_convergence(modules_2)
    epochs_3 = get_convergence(modules_3)
    epochs_4 = get_convergence(modules_4)

    print()
    print('SALINAS - NO ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_2)))
    print('\tstd: {} epochs'.format(np.std(epochs_2)))
    print('\tmax: {} epochs'.format(np.max(epochs_2)))
    print('\tmin: {} epochs'.format(np.min(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_3)))
    print('\tstd: {} epochs'.format(np.std(epochs_3)))
    print('\tmax: {} epochs'.format(np.max(epochs_3)))
    print('\tmin: {} epochs'.format(np.min(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_4)))
    print('\tstd: {} epochs'.format(np.std(epochs_4)))
    print('\tmax: {} epochs'.format(np.max(epochs_4)))
    print('\tmin: {} epochs'.format(np.min(epochs_4)))

    histories = [history for history in
                 glob.iglob('/Users/pablo/Downloads/runners/**/*_training_history.pkl', recursive=True)
                 if 'salinas' in history and 'no_attention' not in history]

    modules_2 = [h for h in histories if '2_modules' in h]
    modules_3 = [h for h in histories if '3_modules' in h]
    modules_4 = [h for h in histories if '4_modules' in h]

    epochs_2 = get_convergence(modules_2)
    epochs_3 = get_convergence(modules_3)
    epochs_4 = get_convergence(modules_4)

    print()
    print('SALINAS - WITH ATTENTION')
    print('\n- 2 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_2)))
    print('\tstd: {} epochs'.format(np.std(epochs_2)))
    print('\tmax: {} epochs'.format(np.max(epochs_2)))
    print('\tmin: {} epochs'.format(np.min(epochs_2)))

    print('\n- 3 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_3)))
    print('\tstd: {} epochs'.format(np.std(epochs_3)))
    print('\tmax: {} epochs'.format(np.max(epochs_3)))
    print('\tmin: {} epochs'.format(np.min(epochs_3)))

    print('\n- 4 modules:\n')

    print('\tmean: {} epochs'.format(np.mean(epochs_4)))
    print('\tstd: {} epochs'.format(np.std(epochs_4)))
    print('\tmax: {} epochs'.format(np.max(epochs_4)))
    print('\tmin: {} epochs'.format(np.min(epochs_4)))

if __name__ == '__main__':
    main()