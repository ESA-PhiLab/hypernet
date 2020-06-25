from ml_intuition.remote_run import queue_build
from time import sleep

datasets = ['houston']
splits = ['grids_v3']
pa = ['0.5']
fold_count = 5
n_classes = 20
base_command = 'python -m scripts.inference_runner {} --n-runs {} ' \
                      '--models-path {} --n-classes {} --batch-size 32768 ' \
                      '--stratified {} --val-size 0.1 --train-size {} --dest-path {} ' \
                      '--channels-idx 2 --spost test --post gaussian --noise-params "{{mean: 0, std: 0.01, pa: {},pb: 1,bc: True}}" ' \
                      ' --experiment-name {} --run-name {}'

if __name__ == '__main__':
    for dataset in datasets:
        for split in splits:
            for p in pa:
                # if split == 'balanced' or split == 'imbalanced':
                    # data = '--data-file-path datasets/{}/{}.npy --ground-truth-path datasets/{}/{}_gt.npy'.format(dataset, dataset, dataset, dataset)
                    # n_runs = 30
                    # models_path = '{}/'.format(dataset) + split
                    # stratified = True if split == 'balanced' else False
                    # # train_size = '30 --train-size 250 --train-size 250 --train-size 150 --train-size 250 --train-size 250 --train-size 20 --train-size 250 ' \
                    # #              '--train-size 15 --train-size 250 --train-size 250 --train-size 250 --train-size 150 --train-size 250 --train-size 50 --train-size 50' if split == 'balanced' else 2715
                    # train_size = 500 if split == 'balanced' else 10000
                    # dest_path = '{}/noise/gaussian/test/pa-'.format(dataset) + str(p) + "/" + str(split)
                    # run_name = str(split) + "_gaussian_test_pa{}".format(p).replace('.', '')
                    # command_run = base_command.format(data, n_runs, models_path, n_classes, stratified, train_size, dest_path, p, dataset, run_name)
                    # command_prebuild = 'dvc pull datasets/{}/{}.npy.dvc datasets/{}/{}_gt.npy.dvc'.format(dataset, dataset, dataset, dataset)
                    # print(command_run)
                    # queue_build(command_prebuild, command_run)
                    # sleep(3)
                if 'grid' in split:
                    for i in range(3):
                        data = '--data-file-path datasets/{}/{}/fold_{}.md5 '.format(dataset, split, i)
                        n_runs = 5
                        models_path = '{}/'.format(dataset) + split + '_fold{}'.format(i)
                        stratified = True
                        train_size = 0
                        dest_path = '{}/noise/gaussian/test/pa-'.format(dataset) + str(p) + "/" + split + '_fold{}'.format(i)
                        run_name = split + '_fold{}'.format(i)+ "_gaussian_test_pa{}".format(p).replace('.', '')
                        command_run = base_command.format(data, n_runs, models_path, n_classes,
                                                          stratified, train_size,
                                                          dest_path, p, dataset, run_name)
                        command_prebuild = 'dvc pull datasets/{}/{}'.format(dataset, split)
                        print(command_run)
                        queue_build(command_prebuild, command_run)
                        sleep(3)
