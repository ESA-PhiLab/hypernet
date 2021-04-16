""" Evaluate the models for cloud detection. """

from typing import Tuple, List, Dict
from pathlib import Path
import numpy as np
from mlflow import log_metrics, log_param
from tensorflow import keras

from cloud_detection.evaluate_38Cloud import evaluate_model as test_38Cloud
from cloud_detection.evaluate_L8CCA import evaluate_model as test_L8CCA


def evaluate_model(
    dataset_name: str,
    model: keras.Model,
    thr: float,
    dpath: Path,
    rpath: Path,
    vids: Tuple[str],
    batch_size: int,
    img_ids: List[str],
    snow_imgs: List[str],
    mlflow: bool,
    gtpath: Path = None,
    vpath: Path = None,
    bands: Tuple[int] = (4, 3, 2, 5),
    bands_names: Tuple[str] = ("red", "green", "blue", "nir"),
    resize: bool = False,
    normalize: bool = True,
    standardize: bool = False
) -> Tuple[Dict, List]:
    """
    Evaluate given dataset.

    :param dataset_name: name of the dataset, one of "38Cloud" and "L8CCA".
    :param model: trained model to make predictions.
    :param thr: threshold to be used during evaluation.
    :param dpath: path to dataset.
    :param rpath: path to directory where results and artifacts
                  should be logged.
    :param vids: tuple of ids of images which should be used
                 to create visualisations. If contains '*' visualisations
                 will be created for all images in the dataset.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param img_ids: if given, process only these images.
    :param snow_imgs: list of snow images IDs.
    :param mlflow: whether to use mlflow
    :param gtpath: path to dataset ground truths.
    :param vpath: path to dataset (false color) visualisation images.
    :param bands: band numbers to load
    :param bands_names: names of the bands to load. Should have the same number
                        of elements as bands.
    :param resize: whether to resize loaded img to gt.
    :param normalize: whether to normalize the image
                      (works only if dataset_name=L8CCA).
    :param standardize: whether to standardize the image
                        (works only if dataset_name=L8CCA).
    :return: mean metrics for normal and snow datasets
             as well as evaluation times for scenes.
    """
    if dataset_name == "38Cloud":
        metrics, scene_times = test_38Cloud(
            model=model,
            thr=thr,
            dpath=dpath,
            gtpath=gtpath,
            vpath=vpath,
            rpath=rpath,
            vids=vids,
            batch_size=batch_size,
            img_ids=img_ids
        )
    elif dataset_name == "L8CCA":
        metrics, scene_times = test_L8CCA(
            model=model,
            thr=thr,
            dpath=dpath,
            rpath=rpath,
            vids=vids,
            batch_size=batch_size,
            bands=bands,
            bands_names=bands_names,
            img_ids=img_ids,
            resize=resize,
            normalize=normalize,
            standardize=standardize
        )
    mean_metrics = {}
    mean_metrics_snow = {}
    for key, value in metrics.items():
        mean_metrics[key] = np.mean(list(value.values()))
        mean_metrics_snow[f"snow_{key}"] = np.mean(
            [value[x] for x in snow_imgs]
        )
    if mlflow:
        log_metrics(mean_metrics)
        log_metrics(mean_metrics_snow)
        log_param(
            f"{dataset_name}_avg_scene_eval_times",
            sum(scene_times) / len(scene_times)
            )
    return mean_metrics, mean_metrics_snow, scene_times
