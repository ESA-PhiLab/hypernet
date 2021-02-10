"""
Evaluate the dataset using and ensemble for the unmixing problem.
"""
from typing import Dict

import numpy as np

from ml_intuition import enums
from ml_intuition.data import io
from ml_intuition.data.utils import get_central_pixel_spectrum
from ml_intuition.evaluation.performance_metrics import \
    calculate_unmixing_metrics
from ml_intuition.evaluation.time_metrics import timeit
from ml_intuition.models import Ensemble


def evaluate(*,
             y_pred: np.ndarray,
             data: Dict,
             dest_path: str,
             endmembers_path: str = None,
             neighborhood_size: int = None):
    ensemble = Ensemble(voting='unmixing')
    vote = timeit(ensemble.vote)
    y_pred, voting_time = vote(y_pred)
    model_metrics = calculate_unmixing_metrics(**{
        'endmembers': np.load(endmembers_path)
        if endmembers_path is not None else None,
        'y_pred': y_pred,
        'y_true': data[enums.Dataset.TEST][enums.Dataset.LABELS],
        'x_true': get_central_pixel_spectrum(
            data[enums.Dataset.TEST][enums.Dataset.DATA],
            neighborhood_size)
    })
    model_metrics['inference_time'] = [voting_time]
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
