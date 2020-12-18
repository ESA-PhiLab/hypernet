"""
Freeze model, quantize it and evaluate N times.
"""

import os
import subprocess

import clize
from clize.parameters import multi
import tensorflow as tf
from scripts import prepare_data, \
    artifacts_reporter
from scripts.quantization import freeze_model, evaluate_graph


def run_experiments(*,
                    input_dir: str,
                    n_runs: int,
                    dest_path: str,
                    data_file_path: str = None,
                    ground_truth_path: str = None,
                    dataset_path: str = None,
                    background_label: int = 0,
                    channels_idx: int = 2,
                    channels_count: int = 103,
                    train_size: ('train_size', multi(min=0)),
                    batch_size: int = 64,
                    stratified: bool = True,
                    gpu: bool = 0):
    """
    Freeze model, quantize it and evaluate N times.

    :param input_dir: Directory with saved data and models, each in separate
        `experiment_n` folder.
    :param n_runs: Number of total experiment runs.
    :param dest_path: Path to where all experiment runs will be saved as
        sub folders in this directory.
    :param data_file_path: Path to the data file. Supported types are: .npy and
        .md5. This is optional, if the data is not already saved in the
        input_dir.
    :param ground_truth_path: Path to the data file.
    :param dataset_path: Path to the already extracted .h5 dataset
    :param background_label: Label indicating the background in GT file
    :param channels_idx: Index specifying the channels position in the provided
        data
    :param channels_count: Number of channels (bands) in the image.
    :param train_size: If float, should be between 0.0 and 1.0.
        If stratified = True, it represents percentage of each class to be extracted,
        If float and stratified = False, it represents percentage of the whole
        dataset to be extracted with samples drawn randomly, regardless of their class.
        If int and stratified = True, it represents number of samples to be
        drawn from each class.
        If int and stratified = False, it represents overall number of samples
        to be drawn regardless of their class, randomly.
        Defaults to 0.8
    :type train_size: Union[int, float]
    :param stratified: Indicated whether the extracted training set should be
        stratified, defaults to True
    :param batch_size: Batch size
    :param gpu: Whether to run quantization on gpu.
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, 'experiment_' + str(experiment_id))
        model_path = os.path.join(input_dir,
                                  'experiment_' + str(experiment_id),
                                  'model_2d')
        created_dataset = False
        if dataset_path is None:
            data_path = os.path.join(input_dir, 'experiment_' + str(experiment_id),
                                     'data.h5')
            created_dataset = True
        else:
            data_path = dataset_path
        os.makedirs(experiment_dest_path, exist_ok=True)

        if not os.path.exists(data_path):
            data_path = os.path.join(experiment_dest_path, 'data.md5')
            prepare_data.main(data_file_path=data_file_path,
                              ground_truth_path=ground_truth_path,
                              output_path=data_path,
                              background_label=background_label,
                              channels_idx=channels_idx,
                              save_data=True,
                              seed=experiment_id,
                              train_size=train_size,
                              stratified=stratified)

        freeze_model.main(model_path=model_path,
                          output_dir=experiment_dest_path)

        node_names_file = os.path.join(experiment_dest_path,
                                       'freeze_input_output_node_name.json')
        frozen_graph_path = os.path.join(experiment_dest_path,
                                         'frozen_graph.pb')
        cmd = 'scripts/quantize.sh ' + node_names_file + ' ' \
              + frozen_graph_path + ' ' + data_path + ' ' + \
              '?,{},1,1'.format(channels_count) + ' ' + \
              'ml_intuition.data.input_fn.calibrate_2d_input' + ' ' + \
              '128' + ' ' + experiment_dest_path + \
              ' ' + str(gpu)
        subprocess.call(cmd, shell=True, env=os.environ.copy())

        graph_path = os.path.join(experiment_dest_path,
                                  'quantize_eval_model.pb')
        evaluate_graph.main(graph_path=graph_path,
                            node_names_path=node_names_file,
                            dataset_path=data_path,
                            batch_size=batch_size)
        if created_dataset:
            os.remove(data_path)

        artifacts_reporter.collect_artifacts_report(experiments_path=dest_path,
                                                    dest_path=dest_path,
                                                    filename='inference_graph_metrics.csv')

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    clize.run(run_experiments)
