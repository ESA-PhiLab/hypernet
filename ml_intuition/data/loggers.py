import json
import re
from typing import Dict

import mlflow
import yaml

from ml_intuition.data.utils import list_to_string
from ml_intuition.enums import MLflowTags, Splits
from scripts.artifacts_reporter import MEAN

LOGGING_EXCLUDED_PARAMS = ['run_name', 'experiment_name', 'use_mlflow',
                           'verbose']


def log_dict_to_mlflow(dict_as_string: str) -> None:
    """
    Log a string which represents a dictionary to MLflow
    :param dict_as_string: A string with dictionary format
    :return: None
    """
    try:
        to_log = json.loads(dict_as_string)
    except Exception:
        to_log = yaml.load(dict_as_string)
    mlflow.log_params(to_log)


def log_params_to_mlflow(args: Dict) -> None:
    """
    Log provided arguments as dictionary to mlflow.
    :param args: Arguments to log
    """
    args['artifacts_storage'] = args.pop('dest_path')
    for arg in args.keys():
        if arg not in LOGGING_EXCLUDED_PARAMS and args[arg] is not None:
            if type(args[arg]) is list:
                args[arg] = list_to_string(args[arg])
                if args[arg] == "":
                    continue
            elif arg == 'noise_params':
                log_dict_to_mlflow(args[arg])
                continue
            mlflow.log_param(arg, args[arg])


def log_tags_to_mlflow(args: Dict):
    """
    Log tags to mlflow based on provided args
    :param args: Argument of the running script
    :return: None
    """
    run_name = args['run_name']
    if Splits.IMBALANCED in run_name:
        mlflow.set_tag(MLflowTags.SPLIT, Splits.IMBALANCED)
    elif Splits.BALANCED in run_name:
        mlflow.set_tag(MLflowTags.SPLIT, Splits.BALANCED)
    elif Splits.GRIDS in run_name:
        # match 'grids' or 'grids_v#' where # is a grid version number
        split = re.findall('{}(?:_v)?[0-9]?'.format(Splits.GRIDS), run_name)[0]
        # get the fold number after the 'fold' keyword,
        # with an optional '_' in between
        fold_id = re.findall(r'fold[_]?(\d+)', run_name)[0]

        mlflow.set_tag(MLflowTags.SPLIT, split)
        mlflow.set_tag(MLflowTags.FOLD, fold_id)

    if MLflowTags.QUANTIZED in run_name:
        mlflow.set_tag(MLflowTags.QUANTIZED, '1')


def log_metrics_to_mlflow(metrics: Dict[str, float], fair: bool = False):
    """
    Log provided metrics to mlflow
    :param metrics: Metrics in a dictionary
    :param fair: Whether to add '_fair' suffix to the metrics name
    :return: None
    """
    for metric in metrics.keys():
        if metric != 'Stats':
            if fair:
                mlflow.log_metric(metric + '_fair', metrics[metric][MEAN])
            else:
                mlflow.log_metric(metric, metrics[metric][MEAN])
