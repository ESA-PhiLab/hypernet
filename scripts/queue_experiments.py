from scripts.remote_run import queue_build


if __name__ == '__main__':
    command_run = 'python -m scripts.inference_runner --data-file-path datasets/houston/houston.npy ' \
                  '--ground-truth-path datasets/houston/houston_gt.npy --n-runs 1 ' \
                  '--models-path houston/balanced --n-classes 20 ' \
                  '--stratified True --val-size 0.1 --train-size 500 --dest-path houston/noise/gaussian/pa-0.1/balanced ' \
                  '--channels-idx 2 --spost test --post gaussian --noise-params "{\"mean\":0,\"std\":0.01,\"pa\":0.2,\"pb\":1,\"bc\":true}" ' \
                  ' --experiment-name houston --run-name balanced_gaussian_test_pa01'
    command_prebuild = 'dvc pull datasets/houston/houston.npy.dvc datasets/houston/houston_gt.npy.dvc'

    queue_build(command_prebuild, command_run)
