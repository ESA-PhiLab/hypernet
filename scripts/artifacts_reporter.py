import collections
import csv
import os

import clize
import numpy as np

from ml_intuition.data import io


EXTENSION = 1


def collect_artifacts_report(*, experiments_path: str, dest_path: str):
    """
    Collect the artifacts report based on the experiment runs
    placed in the "experiments_path" directory.

    :param experiments_path: Path to the directory containing the
        experiment subdirectories.
    :param dest_path: Path to the destination directory or
        full path to the report .csv file.
    """
    all_metrics = io.load_metrics(experiments_path)
    metric_keys = set(tuple(metric_keys)
                      for metric_keys in all_metrics['metric_keys'])
    assert len(metric_keys) == 1, \
        'The metric names should be consistent across all experiment runs.'
    artifacts = {metric_key: [] for metric_key in next(iter(metric_keys))}

    for metric_values in all_metrics['metric_values']:
        for metric_key, metric_value in zip(artifacts.keys(), metric_values):
            artifacts[metric_key].append(float(metric_value))

    stat_report = {'Stats': ['mean', 'std', 'min', 'max']}
    for key in artifacts.keys():
        stat_report[key] = [
            np.mean(artifacts[key]), np.std(artifacts[key]),
            np.min(artifacts[key]), np.max(artifacts[key])
        ]
    if len(os.path.splitext(dest_path)[EXTENSION]) != 0:
        io.save_metrics(dest_path, stat_report)
    else:
        os.makedirs(dest_path, exist_ok=True)
        io.save_metrics(dest_path, stat_report, 'report.csv')


if __name__ == '__main__':
    clize.run(collect_artifacts_report)
