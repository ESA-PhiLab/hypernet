"""
Run experiments given set of hyperparameters.
"""

import os
import shutil

import clize
import mlflow
import tensorflow as tf
from clize.parameters import multi

from ml_intuition import enums
from ml_intuition.data.loggers import log_params_to_mlflow, log_tags_to_mlflow
from ml_intuition.data.utils import parse_train_size, subsample_test_set
from scripts import prepare_data, train_unmixing, evaluate_unmixing, \
    artifacts_reporter
from ml_intuition.models import unmixing_pixel_based_dcae, \
    unmixing_pixel_based_cnn, \
    unmixing_cube_based_cnn, \
    unmixing_cube_based_dcae

# Literature hyperparameters settings:
NEIGHBORHOOD_SIZES = {
    unmixing_cube_based_dcae.__name__: 5,
    unmixing_cube_based_cnn.__name__: 3
}

LEARNING_RATES = {
    unmixing_pixel_based_dcae.__name__: 0.001,
    unmixing_cube_based_dcae.__name__: 0.0005,

    unmixing_pixel_based_cnn.__name__: 0.01,
    unmixing_cube_based_cnn.__name__: 0.001
}


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str = None,
                    train_size: ('train_size', multi(min=0)),
                    val_size: float = 0.1,
                    sub_test_size: int = None,
                    channels_idx: int = -1,
                    neighborhood_size: int = None,
                    n_runs: int = 1,
                    model_name: str,
                    save_data: bool = 0,
                    dest_path: str = None,
                    sample_size: int,
                    n_classes: int,
                    lr: float = None,
                    batch_size: int = 256,
                    epochs: int = 100,
                    verbose: int = 2,
                    shuffle: bool = True,
                    patience: int = 15,
                    use_mlflow: bool = False,
                    endmembers_path: str = None,
                    experiment_name: str = None,
                    run_name: str = None):
    """
    Function for running experiments given a set of hyper parameters.
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the ground-truth data file.
    :param train_size: If float, should be between 0.0 and 1.0,
        if stratified = True, it represents percentage of each class
        to be extracted.
        If float and stratified = False, it represents percentage of the
            whole dataset to be extracted with samples drawn randomly,
            regardless of their class.
         If int and stratified = True, it represents number of samples
            to be drawn from each class.
         If int and stratified = False, it represents overall number of
            samples to be drawn regardless of their class, randomly.
         Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of each class from the training set to be
        extracted as a validation set, defaults to 0.1
    :param sub_test_size: Number of pixels to subsample the test set
        instead of performing the inference on all untrained samples.
    :param channels_idx: Index specifying the channels position in the provided
                         data.
    :param neighborhood_size: Size of the spatial patch.
    :param save_data: Whether to save the prepared dataset
    :param n_runs: Number of total experiment runs.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param lr: Learning rate for the model, i.e., regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle dataset
     dataset_key each epoch.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param use_mlflow: Whether to log metrics and artifacts to mlflow.
    :param endmembers_path: Path to the endmembers file containing
        average reflectances for each class.
        Used only when use_unmixing is true.
    :param experiment_name: Name of the experiment. Used only if
        use_mlflow = True
    :param run_name: Name of the run. Used only if use_mlflow = True.
    """
    if use_mlflow:
        args = locals()
        mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)
        log_params_to_mlflow(args)
        log_tags_to_mlflow(args['run_name'])

    if dest_path is None:
        dest_path = os.path.join(os.path.curdir, "temp_artifacts")

    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path,
            '{}_{}'.format(enums.Experiment.EXPERIMENT, str(experiment_id)))

        os.makedirs(experiment_dest_path, exist_ok=True)

        # Apply default literature hyperparameters:
        if neighborhood_size is None and model_name in NEIGHBORHOOD_SIZES:
            neighborhood_size = NEIGHBORHOOD_SIZES[model_name]
        if lr is None and model_name in LEARNING_RATES:
            lr = LEARNING_RATES[model_name]

        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 train_size=parse_train_size(train_size),
                                 val_size=val_size,
                                 stratified=False,
                                 background_label=-1,
                                 channels_idx=channels_idx,
                                 neighborhood_size=neighborhood_size,
                                 save_data=save_data,
                                 seed=experiment_id,
                                 use_unmixing=True)
        if sub_test_size is not None:
            subsample_test_set(data[enums.Dataset.TEST], sub_test_size)
        train_unmixing.train(model_name=model_name,
                             dest_path=experiment_dest_path,
                             data=data,
                             sample_size=sample_size,
                             neighborhood_size=neighborhood_size,
                             n_classes=n_classes,
                             lr=lr,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=verbose,
                             shuffle=shuffle,
                             patience=patience,
                             endmembers_path=endmembers_path,
                             seed=experiment_id)

        evaluate_unmixing.evaluate(
            model_path=os.path.join(experiment_dest_path, model_name),
            data=data,
            dest_path=experiment_dest_path,
            neighborhood_size=neighborhood_size,
            batch_size=batch_size,
            endmembers_path=endmembers_path)

        tf.keras.backend.clear_session()

    artifacts_reporter.collect_artifacts_report(
        experiments_path=dest_path,
        dest_path=dest_path,
        use_mlflow=use_mlflow)

    if use_mlflow:
        mlflow.log_artifacts(dest_path, artifact_path=dest_path)
        shutil.rmtree(dest_path)


if __name__ == '__main__':
    clize.run(run_experiments)
