"""
Run inference given set of hyperparameters for the unmixing problem.
"""

import os
import re
import shutil

import clize
import mlflow
import tensorflow as tf

from ml_intuition import enums
from ml_intuition.data.loggers import log_params_to_mlflow, log_tags_to_mlflow
from ml_intuition.data.utils import get_mlflow_artifacts_path, \
    subsample_test_set
from scripts import prepare_data, artifacts_reporter
from scripts.unmixing import evaluate_unmixing


def run_experiments(*,
                    data_file_path: str = None,
                    ground_truth_path: str = None,
                    train_size: int,
                    val_size: float = 0.1,
                    sub_test_size: int,
                    channels_idx: int = 0,
                    neighborhood_size: int = None,
                    save_data: bool = False,
                    n_runs: int = 1,
                    dest_path: str,
                    models_path: str,
                    model_name: str,
                    n_classes: int,
                    use_ensemble: bool = False,
                    ensemble_copies: int = None,
                    voting: str = 'mean',
                    voting_model: str = None,
                    voting_model_params: str = None,
                    batch_size: int = 256,
                    noise_params: str = None,
                    endmembers_path: str = None,
                    use_mlflow: bool = False,
                    experiment_name: str = None,
                    model_exp_name: str = None,
                    run_name: str = None):
    """
    Function for running the inference for the unmixing problem
    given a set of hyperparameters.

    :param data_file_path: Path to the data file. It should be a numpy array.
    :param ground_truth_path: Path to the ground-truth data file.
        It should be a numpy array.
    :param train_size: If float, should be between 0.0 and 1.0,
        if int, it represents number of samples to draw from data.
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of samples to extract from the training set.
    :param sub_test_size: Number of pixels to subsample the test set
        instead of performing the inference on the entire subset.
    :param channels_idx: Index specifying the channels
        position in the provided data.
    :param neighborhood_size: Size of the spatial patch.
    :param save_data: Boolean indicating whether to save the prepared dataset.
    :param n_runs: Number of total experiment runs.
    :param dest_path: Path to the directory where all experiment runs
        will be saved as subdirectories.
    :param models_path: Path to the directory where the previously trained
        models are stored.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param n_classes: Number of classes.
    :param use_ensemble: Boolean indicating whether to use the
        ensemble functionality for prediction.
    :param ensemble_copies: Number of model copies for the ensemble.
    :param voting: Method of ensemble voting. If 'booster',
        employs a new model, which is trained on the
        ensemble predictions on the training set. Else if 'mean', averages
        the predictions of all models, without any weights.
    :param voting_model: Type of the model to use when the voting
        argument is set to 'booster'. This indicates, that a new model
        is trained on the ensemble's predictions on the learning set,
        to leverage the quality of the regression. Supported models are:
        SVR (support vector machine for regression), RFR (random forest
        for regression) and DTR (decision tree for regression).
    :param voting_model_params: Parameters of the voting model.
        Used only when the type of voting is set to 'booster'.
        Should be specified analogously to the noise injection parameters
        in the 'noise' module.
    :param batch_size: Size of the batch used in training phase,
        it is the number of samples to utilize per single gradient step.
    :param noise_params: Parameters for the noise when creating
        copies of the base model. Those can be for instance the mean,
        or standard deviation of the noise.
        For the details see the 'noise' module.
        Exemplary value for this parameter is "{"mean": 0, "std": 1}".
    :param endmembers_path: Path to the endmembers file containing
        the average reflectances for each class. Used only when
        'use_unmixing' is set to True.
    :param use_mlflow: Boolean indicating whether to log metrics
        and artifacts to mlflow.
    :param experiment_name: Name of the experiment. Used only if
        'use_mlflow' is set to True.
    :param model_exp_name: Name of the experiment. Used only if
        'use_mlflow' is set to True.
    :param run_name: Name of the run. Used only if 'use_mlflow' is set to True.
    """
    if use_mlflow:
        args = locals()
        mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        log_params_to_mlflow(args)
        log_tags_to_mlflow(args['run_name'])
        models_path = get_mlflow_artifacts_path(models_path, model_exp_name)

    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, 'experiment_' + str(experiment_id))
        model_name_regex = re.compile('unmixing_.*')
        model_dir = os.path.join(models_path, f'experiment_{experiment_id}')
        model_name = list(filter(model_name_regex.match,
                                 os.listdir(model_dir)))[0]
        model_path = os.path.join(model_dir, model_name)

        os.makedirs(experiment_dest_path, exist_ok=True)

        data_source = prepare_data.main(data_file_path=data_file_path,
                                        ground_truth_path=ground_truth_path,
                                        train_size=train_size,
                                        val_size=val_size,
                                        stratified=False,
                                        background_label=-1,
                                        channels_idx=channels_idx,
                                        neighborhood_size=neighborhood_size,
                                        save_data=save_data,
                                        seed=experiment_id,
                                        use_unmixing=True)
        if sub_test_size is not None:
            subsample_test_set(data_source[enums.Dataset.TEST], sub_test_size)
        evaluate_unmixing.evaluate(
            model_path=model_path,
            data=data_source,
            dest_path=experiment_dest_path,
            use_ensemble=use_ensemble,
            ensemble_copies=ensemble_copies,
            endmembers_path=endmembers_path,
            voting=voting,
            voting_model=voting_model,
            noise_params=noise_params,
            batch_size=batch_size,
            seed=experiment_id,
            neighborhood_size=neighborhood_size,
            voting_model_params=voting_model_params)

        tf.keras.backend.clear_session()

    artifacts_reporter.collect_artifacts_report(experiments_path=dest_path,
                                                dest_path=dest_path,
                                                use_mlflow=use_mlflow)

    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.log_artifacts(dest_path, artifact_path=dest_path)
        shutil.rmtree(dest_path)


if __name__ == '__main__':
    clize.run(run_experiments)
