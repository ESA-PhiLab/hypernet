"""
Run ensemble using N number of different models. Models might use datasets
preprocessed in various ways.
"""

import os
import shutil
import re

import clize
import mlflow
import tensorflow as tf
from clize.parameters import multi

from ml_intuition import enums
from scripts import prepare_data, artifacts_reporter, predict_with_model
from scripts.ensemble import evaluate_with_ensemble
from ml_intuition.enums import Experiment
from ml_intuition.data.io import load_processed_h5
from ml_intuition.data.utils import get_mlflow_artifacts_path, parse_train_size
from ml_intuition.data.loggers import log_params_to_mlflow, log_tags_to_mlflow


def run_experiments(*,
                    data_file_paths: ('d', multi(min=1)),
                    train_size: ('train_size', multi(min=0)),
                    val_size: float = 0.1,
                    stratified: bool = True,
                    background_label: int = 0,
                    channels_idx: int = 0,
                    neighborhood_sizes: ('n', multi(min=1)),
                    save_data: bool = False,
                    n_runs: int,
                    dest_path: str,
                    model_paths: ('m', multi(min=1)),
                    model_experiment_names: ('e', multi(min=1)),
                    n_classes: int,
                    voting: str = 'hard',
                    batch_size: int = 1024,
                    post_noise_sets: ('spost', multi(min=0)),
                    post_noise: ('post', multi(min=0)),
                    noise_params: str = None,
                    use_mlflow: bool = False,
                    experiment_name: str = None,
                    run_name: str = None):
    """
    Function for running experiments given a set of hyper parameters.
    :param data_file_paths: Path to the data file. Supported types are: .npy
    :param train_size: If float, should be between 0.0 and 1.0,
                        if stratified = True, it represents percentage of each
                        class to be extracted,
                 If float and stratified = False, it represents percentage of the
                    whole dataset to be extracted with samples drawn randomly,
                    regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage of
                     each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param stratified: Indicated whether the extracted training set should be
                     stratified, defaults to True
    :param background_label: Label indicating the background in GT file
    :param channels_idx: Index specifying the channels position in the provided
                         data
    :param neighborhood_sizes: List of sizes of neighborhoods of provided models.
    :param save_data: Whether to save the prepared dataset
    :param n_runs: Number of total experiment runs.
    :param dest_path: Path to where all experiment runs will be saved as
        subfolders in this directory.
    :param model_paths: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param model_experiment_names: Names of experiments that provided models
        belong to.
    :param n_classes: Number of classes.
    :param voting: Method of ensemble voting. If ‘hard’, uses predicted class
        labels for majority rule voting. Else if ‘soft’, predicts the class
        label based on the argmax of the sums of the predicted probabilities.
    :param batch_size: Size of the batch for the inference
    :param post_noise_sets: The list of sets to which the noise will be
        injected. One element can either be "train", "val" or "test".
    :param post_noise: The list of names of noise injection methods after
        the normalization transformations.
    :param noise_params: JSON containing the parameter setting of injection methods.
        Exemplary value for this parameter: "{"mean": 0, "std": 1, "pa": 0.1}".
        This JSON should include all parameters for noise injection
        functions that are specified in pre_noise and post_noise arguments.
        For the accurate description of each parameter, please
        refer to the ml_intuition/data/noise.py module.
    :param use_mlflow: Whether to log metrics and artifacts to mlflow.
    :param experiment_name: Name of the experiment. Used only if
        use_mlflow = True
    :param run_name: Name of the run. Used only if use_mlflow = True.
    """
    train_size = parse_train_size(train_size)
    if use_mlflow:
        args = locals()
        mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        log_params_to_mlflow(args)
        log_tags_to_mlflow(args['run_name'])
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, 'experiment_' + str(experiment_id))

        os.makedirs(experiment_dest_path, exist_ok=True)
        models_test_predictions = []
        models_train_predictions = []

        for data_file_path, model_path, model_experiment_name, neighborhood_size in \
                zip(
                    data_file_paths,
                    model_paths,
                    model_experiment_names,
                    neighborhood_sizes):
            model_path = get_mlflow_artifacts_path(model_path,
                                                   model_experiment_name)
            model_name_regex = re.compile('model_.*')
            model_dir = os.path.join(model_path, f'experiment_{experiment_id}')
            model_name = \
                list(filter(model_name_regex.match, os.listdir(model_dir)))[0]
            model_path = os.path.join(model_dir, model_name)
            if data_file_path.endswith(
                    '.h5') and 'patches' not in data_file_path:
                data_source = load_processed_h5(data_file_path=data_file_path)
            else:
                data_source = prepare_data.main(data_file_path=data_file_path,
                                                ground_truth_path='',
                                                train_size=train_size,
                                                val_size=val_size,
                                                stratified=stratified,
                                                background_label=background_label,
                                                channels_idx=channels_idx,
                                                neighborhood_size=int(
                                                    neighborhood_size),
                                                save_data=save_data,
                                                seed=experiment_id)

            test_predictions = predict_with_model.predict(
                model_path=model_path,
                data=data_source,
                batch_size=batch_size,
                dataset_to_predict=enums.Dataset.TEST
            )

            models_test_predictions.append(test_predictions)

            if voting == 'classifier':
                train_predictions = predict_with_model.predict(
                    model_path=model_path,
                    data=data_source,
                    batch_size=batch_size,
                    dataset_to_predict=enums.Dataset.TRAIN
                )
                models_train_predictions.append(train_predictions)
            tf.keras.backend.clear_session()

        evaluate_with_ensemble.evaluate(
            y_pred=models_test_predictions,
            model_path=model_path,
            data=data_source,
            dest_path=experiment_dest_path,
            voting=voting,
            train_set_predictions=models_train_predictions)

    artifacts_reporter.collect_artifacts_report(experiments_path=dest_path,
                                                dest_path=dest_path,
                                                use_mlflow=use_mlflow)
    fair_report_path = os.path.join(dest_path, Experiment.REPORT_FAIR)
    artifacts_reporter.collect_artifacts_report(experiments_path=dest_path,
                                                dest_path=fair_report_path,
                                                filename=Experiment.INFERENCE_FAIR_METRICS,
                                                use_mlflow=use_mlflow)
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.log_artifacts(dest_path, artifact_path=dest_path)
        shutil.rmtree(dest_path)


if __name__ == '__main__':
    clize.run(run_experiments)
