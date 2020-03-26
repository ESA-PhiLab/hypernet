import collections
import csv
import glob
import os

import clize
import numpy as np


def collect_artifacts(*,
                      experiments_path: str):
    all_metrics = {'metric_keys': [], 'metric_values': []}
    for experiment_dir in glob.glob(os.path.join(experiments_path,
                                                 'experiment*')):
        with open(os.path.join(experiment_dir,
                               'inference_metrics.csv')) as metric_file:
            reader = csv.reader(metric_file, delimiter=',')
            for row, key in zip(reader, all_metrics.keys()):
                all_metrics[key].append(row)

    metric_keys = set(tuple(metric_keys)
                      for metric_keys in all_metrics['metric_keys'])
    assert len(metric_keys) == 1, \
        'The metric names should be consistent across all experiments.'
    artifacts = {metric_key: [] for metric_key in next(iter(metric_keys))}

    for metric_values in all_metrics['metric_values']:
        for metric_key, metric_value in zip(artifacts.keys(), metric_values):
            artifacts[metric_key].append(float(metric_value))

    stat_report = {'Stats': ['mean', 'std', 'min', 'max']}
    for key in artifacts.keys():
        stat_report[key] = [
            np.mean(artifacts[key]), np.std(artifacts[key]),
            np.min(artifacts[key]), np.max(artifacts[key])]

    with open(os.path.join(experiments_path, 'metrics.csv'), 'w') as file:
        write = csv.writer(file)
        write.writerow(stat_report.keys())
        write.writerows(zip(*stat_report.values()))


if __name__ == '__main__':
    clize.run(collect_artifacts)
