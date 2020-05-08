"""
Freeze model, quantize it and evaluate for multiple runs.
"""

import os
import subprocess

import clize
import tensorflow as tf
from scripts import evaluate_graph, freeze_model
from ml_intuition.data.io import load_processed_h5


def run_experiments(*,
                    input_dir: str,
                    n_runs: int,
                    dest_path: str,
                    gpu: bool = 0):
    """
    Function for running experiments given a set of hyperparameters.
    :param input_dir: Directory with saved data and models, each in separate
        `experiment_n` folder.
    :param n_runs: Number of total experiment runs.
    :param dest_path: Path to where all experiment runs will be saved as
        subfolders in this directory.
    :param gpu: Whether to run quantization on gpu.

    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, 'experiment_' + str(experiment_id))
        model_path = os.path.join(input_dir,
                                  'experiment_' + str(experiment_id),
                                  'model_2d')
        data_path = os.path.join(input_dir, 'experiment_' + str(experiment_id),
                                 'data.h5')
        os.makedirs(experiment_dest_path, exist_ok=True)
        data = load_processed_h5(data_file_path=data_path)

        freeze_model.main(model_path=model_path,
                          output_dir=experiment_dest_path)

        node_names_file = os.path.join(experiment_dest_path,
                                       'freeze_input_output_node_name.json')
        frozen_graph_path = os.path.join(experiment_dest_path,
                                         'frozen_graph.pb')
        cmd = '../quantize.sh ' + node_names_file + ' ' \
              + frozen_graph_path + ' ' + data_path + ' ' + '?,103,1,1' + ' ' + \
              'ml_intuition.data.input_fn.calibrate_2d_input' + ' ' + '64' + ' ' \
              + experiment_dest_path + ' ' + str(gpu)
        subprocess.call(cmd, shell=True, env=os.environ.copy())

        graph_path = os.path.join(experiment_dest_path,
                                  'quantize_eval_model.pb')
        evaluate_graph.main(graph_path=graph_path,
                            node_names_path=node_names_file,
                            dataset_path=data_path)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    clize.run(run_experiments)
