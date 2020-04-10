import collections
import csv
import os

import clize
import numpy as np

from ml_intuition.data import io


def collect_artifacts_report(*, experiments_path: str, dest_path: str):
    """
    Collect the artifacts report based on the experiment runs
    placed in the "experiments_path" directory.

    :param experiments_path: Path to the directory containing the
        experiment subdirectories.
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
    os.makedirs(dest_path, exist_ok=True)
    io.save_metrics(dest_path, 'report.csv', stat_report)


if __name__ == '__main__':
    clize.run(collect_artifacts_report)
