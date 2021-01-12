import clize
import pandas as pd

from ml_intuition.data.io import save_ml_report
from scripts.multispectral.train_classifier import train_and_eval_classifier
from scripts.multispectral.train_regression import train_and_eval_regression


def run_experiments(*,
                    dataframe_path: str,
                    label_name: str,
                    output_dir_path: str,
                    model_name: str,
                    train_fraction: float,
                    seed: int,
                    verbose: int,
                    n_jobs: int) -> None:
    """
    Function for running experiments on multispectral datasets.

    :param dataframe_path: Dataframe containing all samples as a design matrix.
        The rows indicate observations, whereas the columns
        indicate explanatory variables.
    :param label_name: The name of the dependent variable in the dataframe.
        All other columns are assumed to serve as input features and the
        regression and classification models are build on top of them.
    :param output_dir_path: Path to the destination output directory.
    :param model_name: Type of the model used for the experiments.
    :param train_fraction: Fraction of the samples employed
        for training the models. For classification problem,
        the division is stratified to preserve the original
        distribution of classes.
    :param seed: Seed used for the experiments reproduction.
    :param verbose: Verbosity mode.
    :param n_jobs: Number of jobs for parallel computing.
    :return: None.
    """
    dataframe = pd.read_csv(dataframe_path)
    if model_name.split('_')[-1] == 'reg':
        test_report, best_params = train_and_eval_regression(
            dataframe=dataframe,
            label_name=label_name,
            train_fraction=train_fraction,
            model_name=model_name,
            seed=seed,
            verbose=verbose,
            n_jobs=n_jobs)
    else:
        test_report, best_params = train_and_eval_classifier(
            dataframe=dataframe,
            label_name=label_name,
            train_fraction=train_fraction,
            model_name=model_name,
            seed=seed,
            verbose=verbose,
            n_jobs=n_jobs)

    save_ml_report(output_dir_path=output_dir_path,
                   model_name=model_name,
                   test_report=test_report,
                   best_params=best_params,
                   train_fraction=train_fraction)


if __name__ == '__main__':
    clize.run(run_experiments)
