"""
Run experiments given set of hyperparameters.
"""

import os

import clize
import tensorflow as tf
from clize.parameters import multi
from scripts import evaluate_model, prepare_data


def run_experiments(*,
                    data_file_path: str,
                    ground_truth_path: str,
                    train_size: float = 0.8,
                    val_size: float = 0.1,
                    stratified: bool = True,
                    background_label: int = 0,
                    channels_idx: int = 0,
                    save_data: bool = False,
                    n_runs: int,
                    dest_path: str,
                    models_path: str,
                    n_classes: int,
                    verbose: int = 2,
                    pre_noise_sets: ('spre', multi(min=0)),
                    post_noise: ('post', multi(min=0)),
                    noise_params: str = None):
    """
    Function for running experiments given a set of hyperparameters.
    :param data_file_path: Path to the data file. Supported types are: .npy
    :param ground_truth_path: Path to the ground-truth data file.
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
    :param save_data: Whether to save the prepared dataset
    :param n_runs: Number of total experiment runs.
    :param dest_path: Path to where all experiment runs will be saved as
        subfolders in this directory.
    :param models_path: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param n_classes: Number of classes.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param pre_noise_sets: The list of sets to which the noise will be
        injected. One element can either be "train", "val" or "test".
    :param post_noise: The list of names of noise injection methods after
        the normalization transformations.
    :param noise_params: JSON containing the parameter setting of injection methods.
        Examplary value for this parameter: "{"mean": 0, "std": 1, "pa": 0.1}".
        This JSON should include all parameters for noise injection
        functions that are specified in pre_noise and post_noise arguments.
        For the accurate description of each parameter, please
        refer to the ml_intuition/data/noise.py module.
    """
    for experiment_id in range(n_runs):
        experiment_dest_path = os.path.join(
            dest_path, 'experiment_' + str(experiment_id))
        model_path = os.path.join(models_path,
                                  'experiment_' + str(experiment_id),
                                  'model_2d')
        if save_data:
            data_source = os.path.join(experiment_dest_path, 'data.h5')
        else:
            data_source = None
        os.makedirs(experiment_dest_path, exist_ok=True)

        data = prepare_data.main(data_file_path=data_file_path,
                                 ground_truth_path=ground_truth_path,
                                 output_path=data_source,
                                 train_size=train_size,
                                 val_size=val_size,
                                 stratified=stratified,
                                 background_label=background_label,
                                 channels_idx=channels_idx,
                                 save_data=save_data,
                                 seed=experiment_id)

        if not save_data:
            data_source = data

        evaluate_model.evaluate(
            model_path=model_path,
            data=data_source,
            dest_path=experiment_dest_path,
            verbose=verbose,
            n_classes=n_classes,
            noise=post_noise,
            noise_sets=pre_noise_sets,
            noise_params=noise_params)

        tf.keras.backend.clear_session()


if __name__ == '__main__':
    clize.run(run_experiments)
