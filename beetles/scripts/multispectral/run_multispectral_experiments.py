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
        indicate explanatory variables. One of those columns is the
        target or dependent variable which the model learns.
        The target variable for the classification should be in the raw nominal
        form (simply put as strings, as the one-hot-encoding is done already
        in the pipeline on this very column), whereas for the regression problem,
        the numerical form is required (standard regression target).
    :param label_name: The name of the dependent variable in the dataframe.
        All other columns are assumed to serve as input features and the
        regression and classification models are build on top of them.
        In other words, the dataset should only consist of one dependent
        variable column, and other features i.e., explanatory variables.
    :param output_dir_path: Path to the destination output directory.
    :param model_name: Type of the model used for the experiments.
        For the classification task, the name should end with "_clf" suffix,
        whereas for the regression problem it is simply "_reg".
        For example to employ the decision tree classifier one
        should specify the model_name as a "decision_tree_clf".
    :param train_fraction: Fraction of the samples employed
        for training the models. For classification problem,
        the division is stratified to preserve the original
        distribution of classes. For the regression task, the data sampling
        is not stratified, and random subsets are generated.
    :param seed: Seed used for the experiments reproduction.
    :param verbose: Verbosity mode.
    :param n_jobs: Number of jobs utilized for the parallel computing.
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
