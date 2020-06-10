from scripts.remote_run import queue_build
from time import sleep

splits = ['balanced', 'imbalanced', 'grids_v2', 'grids_v3']
# pa = ['0.1']
pa = ['0.1', '0.2', '0.3', '0.4', '0.5']
fold_count = 5
base_command = 'python -m scripts.inference_runner {} --n-runs {} ' \
                      '--models-path {} --n-classes 20 --batch-size 32768 ' \
                      '--stratified {} --val-size 0.1 --train-size {} --dest-path {} ' \
                      '--channels-idx 2 --spost test --post gaussian --noise-params "{{mean: 0,std: 0.01,pa: {},pb: 1,bc: True}}" ' \
                      ' --experiment-name houston --run-name {}'

# command_run = 'python -m scripts.inference_runner --data-file-path datasets/houston/houston.npy ' \
#               '--ground-truth-path datasets/houston/houston_gt.npy --n-runs 1 ' \
#               '--models-path houston/balanced --n-classes 20 --batch-size 16384 ' \
#               '--stratified True --val-size 0.1 --train-size 500 --dest-path houston/noise/gaussian/pa-0.1/balanced ' \
#               '--channels-idx 2 --spost test --post gaussian --noise-params {} ' \
#               ' --experiment-name houston --run-name balanced_gaussian_test_pa01'

if __name__ == '__main__':
    for split in splits:
        for p in pa:
            # if split == 'balanced' or split == 'imbalanced':
            #     data = '--data-file-path datasets/houston/houston.npy --ground-truth-path datasets/houston/houston_gt.npy'
            #     n_runs = 30
            #     models_path = 'houston/' + split
            #     stratified = True if split == 'balanced' else False
            #     train_size = 500 if split == 'balanced' else 10000
            #     dest_path = 'houston/noise/gaussian/pa-' +str(p) + "/" + str(split)
            #     run_name = str(split) + "_gaussian_test_pa{}".format(p).replace('.', '')
            #     command_run = base_command.format(data, n_runs, models_path, stratified, train_size, dest_path, p, run_name)
            #     command_prebuild = 'dvc pull datasets/houston/houston.npy.dvc datasets/houston/houston_gt.npy.dvc'
            #     print(command_run)
            #     queue_build(command_prebuild, command_run)
            #     sleep(3)
            if 'grid' in split:
                for i in range(fold_count):
                    data = '--data-file-path datasets/houston/{}/fold_{}.md5 '.format(split, i)
                    n_runs = 5
                    models_path = 'houston/' + split + '_fold{}'.format(i)
                    stratified = True
                    train_size = 0
                    dest_path = 'houston/noise/gaussian/pa-' + str(p) + "/" + split + '_fold{}'.format(i)
                    run_name = split + '_fold{}'.format(i)+ "_gaussian_test_pa{}".format(p).replace('.', '')
                    command_run = base_command.format(data, n_runs, models_path,
                                                      stratified, train_size,
                                                      dest_path, p, run_name)
                    command_prebuild = 'dvc pull datasets/houston/{}'.format(split)
                    print(command_run)
                    queue_build(command_prebuild, command_run)
                    sleep(3)

