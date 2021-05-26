"""
Get evaluation metrics for given model on 38-Cloud test set.

If you plan on using this implementation, please cite our work:
@INPROCEEDINGS{Grabowski2021IGARSS,
author={Grabowski, Bartosz and Ziaja, Maciej and Kawulok, Michal
and Nalepa, Jakub},
booktitle={IGARSS 2021 - 2021 IEEE International Geoscience
and Remote Sensing Symposium},
title={Towards Robust Cloud Detection in
Satellite Images Using U-Nets},
year={2021},
note={in press}}
"""

import os
import re
import time
import uuid
import clize
import numpy as np
import tensorflow as tf
from clize.parameters import multi
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from mlflow import log_metrics, log_artifacts, log_params
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from cloud_detection import losses
from cloud_detection.data_gen import DG_38Cloud
from cloud_detection.validate import (
    make_precision_recall,
    make_roc,
    make_activation_hist,
)
from cloud_detection.utils import (
    unpad,
    get_metrics_tf,
    save_vis,
    setup_mlflow,
    load_image_paths,
    make_paths
)


def get_full_scene_img(path: Path, img_id: str) -> np.ndarray:
    """
    Get image of given id as np.array with values in range 0 to 1.

    :param path: path to the dataset.
    :param img_id: ID of the image.
    :return: image.
    """
    img_path = next(path.glob("*" + img_id + "*"))
    return np.array(load_img(img_path)) / 255


def get_img_pred(
    path: Path, img_id: str, model: keras.Model, batch_size: int,
    patch_size: int = 384
) -> Tuple[np.ndarray, float]:
    """
    Generates prediction for a given image.

    :param path: path containing directories with image channels.
    :param img_id: ID of the considered image.
    :param model: trained model to make predictions.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param patch_size: size of the image patches.
    :return: prediction for the given image along with evaluation time.
    """
    (test_files,) = load_image_paths(
        base_path=path, split_ratios=[1.0], shuffle=False, img_id=img_id
    )
    testgen = DG_38Cloud(
        files=test_files, batch_size=batch_size, shuffle=False, with_gt=False
    )
    tbeg = time.time()
    pred = model.predict_generator(testgen)
    scene_time = time.time() - tbeg
    print(f"Scene prediction took { scene_time } seconds")

    img_shape = get_img_pred_shape(test_files, patch_size)
    img = np.full(img_shape, np.inf)
    for i, fnames in enumerate(test_files):
        red_fname = str(fnames["red"])
        row, col = re.search("([0-9]*)_by_([0-9]*)", red_fname).groups()
        row, col = int(row), int(col)
        img[
            (row - 1) * patch_size: row * patch_size,
            (col - 1) * patch_size: col * patch_size,
        ] = pred[i]
    return img, scene_time


def get_img_pred_shape(files: List[Dict[str, Path]], patch_size: int) -> Tuple:
    """
    Infers shape of the predictions of the considered image.

    :param files: paths to patch files;
          structured as: list_of_files['file_channel', Path].
    :param patch_size: size of the image patches.
    :return: shape of the predictions of the considered image.
    """
    row_max, col_max = 0, 0
    for fnames in files:
        red_fname = str(fnames["red"])
        row, col = re.search("([0-9]*)_by_([0-9]*)", red_fname).groups()
        row, col = int(row), int(col)
        row_max = max(row_max, row)
        col_max = max(col_max, col)
    return patch_size * row_max, patch_size * col_max, 1


def load_img_gt(path: Path, fname: str) -> np.ndarray:
    """
    Load image ground truth.

    :param path: path containing image gts.
    :param fname: image gt file name.
    :return: image ground truth.
    """
    img = np.array(load_img(path / fname, color_mode="grayscale"))
    return np.expand_dims(img / 255, axis=-1)


def evaluate_model(
    model: keras.Model,
    thr: float,
    dpath: Path,
    gtpath: Path,
    vpath: Path,
    rpath: Path,
    vids: Tuple[str],
    batch_size: int,
    img_ids: List[str] = None,
    metric_fns: List[Callable] = [normalized_mutual_info_score,
                                  adjusted_rand_score],
    mlflow: bool = False,
    run_name: str = None,
) -> Tuple[Dict, List]:
    """
    Get evaluation metrics for given model on 38-Cloud testset.

    :param model: trained model to make predictions.
    :param thr: threshold to be used during evaluation.
    :param dpath: path to dataset.
    :param gtpath: path to dataset ground truths.
    :param vpath: path to dataset (false color) visualisation images.
    :param rpath: path to directory where results and artifacts
                  should be logged.
    :param vids: tuple of ids of images which should be used
                 to create visualisations. If contains '*' visualisations
                 will be created for all images in the dataset.
    :param batch_size: size of generated batches, only one batch is loaded
           to memory at a time.
    :param img_ids: if given, process only these images.
    :param metric_fns: non-Tensorflow metric functions to run evaluation
                       of the model. Must be of the form
                       func(labels_true, labels_pred).
    :param mlflow: whether to use MLFlow.
    :param run_name: name of the run.
    :return: evaluation metrics and evaluation times for scenes.
    """
    Path(rpath).mkdir(parents=True, exist_ok=False)
    if mlflow:
        setup_mlflow(run_name)
        params = dict(locals())
        del params["model"]
        del params["metric_fns"]
        log_params(params)
    metrics = {}
    scene_times = []
    for metric_fn in model.metrics:
        if type(metric_fn) is str:
            metric_name = metric_fn
        else:
            metric_name = metric_fn.__name__
        metrics[f"38Cloud_{metric_name}"] = {}
    for metric_fn in metric_fns:
        metrics[f"38Cloud_{metric_fn.__name__}"] = {}

    for fname in os.listdir(gtpath):
        img_id = fname[fname.find("LC08"): fname.find(".TIF")]
        if img_ids is not None:
            if img_id not in img_ids:
                continue
        print(f"Processing {img_id}", flush=True)
        img_pred, scene_time = get_img_pred(dpath, img_id, model, batch_size)
        scene_times.append(scene_time)
        img_gt = load_img_gt(gtpath, fname)
        img_pred = unpad(img_pred, img_gt.shape)
        img_metrics = get_metrics_tf(img_gt, img_pred > thr, model.metrics)
        for metric_fn in model.metrics:
            if type(metric_fn) is str:
                metric_name = metric_fn
            else:
                metric_name = metric_fn.__name__
            metrics[f"38Cloud_{metric_name}"][img_id] = img_metrics[
                f"{metric_name}"
            ]
        for metric_fn in metric_fns:
            metrics[f"38Cloud_{metric_fn.__name__}"][img_id] = metric_fn(
                img_gt.reshape(-1),
                (img_pred > thr).reshape(-1)
            )
        print(
            "Average inference time:" +
            f"{sum(scene_times) / len(scene_times)} seconds"
        )

        if img_id in vids or "*" in vids:
            print(f"Creating visualisation for {img_id}")
            img_vis = 0.7 * get_full_scene_img(vpath, img_id)
            save_vis(img_id, img_vis, img_pred > thr, img_gt, rpath)

        if img_metrics["jaccard_index_metric"] < 0.6:
            print(f"Will make insights for {img_id}", flush=True)
            y_gt = img_gt.ravel()
            y_pred = np.round(img_pred.ravel(), decimals=5)

            make_roc(y_gt, y_pred, rpath / img_id, thr_marker=thr)
            make_precision_recall(
                y_gt, y_pred, rpath / img_id, thr_marker=thr)

            # Make histogram with more rounded predictions
            # for performance reasons
            y_pred = np.round(y_pred, decimals=2)
            make_activation_hist(y_pred, rpath / img_id)

    return metrics, scene_times


def run_evaluation(
    *,
    model_hash: str,
    thr: float = 0.5,
    dpath: str = "datasets/clouds/38-Cloud/38-Cloud_test",
    gtpath: str = "datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts",
    vpath: str = "datasets/clouds/38-Cloud/38-Cloud_test/Natural_False_Color",
    rpath: str = "artifacts/",
    vids: ("v", multi(min=1)),
    batch_size: int = 32,
    img_ids: ("iid", multi(min=0)),
    mlflow: bool = False,
    run_name: str = None,
):
    """
    Load model given model hash and get evaluation metrics on 38-Cloud testset.

    :param model_hash: MLFlow hash of the model to load.
    :param thr: threshold to be used during evaluation.
    :param dpath: path to dataset.
    :param gtpath: path to dataset ground truths.
    :param vpath: path to dataset (false color) visualisation images.
    :param rpath: path to directory where results
                  and artifacts should be logged.
    :param vids: tuple of ids of images which should be used to create
                 visualisations. If contains '*' visualisations will be
                 created for all images in the dataset.
    :param batch_size: size of generated batches, only one batch is loaded
           to memory at a time.
    :param img_ids: if given, process only these images.
    :param mlflow: whether to use MLFlow.
    :param run_name: name of the run.
    """
    snow_imgs = [
        "LC08_L1TP_064015_20160420_20170223_01_T1",
        "LC08_L1TP_035035_20160120_20170224_01_T1",
        "LC08_L1TP_050024_20160520_20170324_01_T1",
    ]
    if img_ids == []:
        img_ids = None
    else:
        snow_imgs = list(set(snow_imgs) & set(img_ids))
    dpath, gtpath, vpath, rpath = make_paths(dpath, gtpath, vpath, rpath)
    rpath = rpath / uuid.uuid4().hex
    print(f"Working dir: {os.getcwd()}, "
          + f"artifacts dir: {rpath}",
          flush=True)
    mpath = Path(
        f"/media/ML/mlflow/beetle/artifacts/34/{model_hash}/"
        # Change init_model to model for old models
        + "artifacts/init_model/data/model.h5"
    )
    model = keras.models.load_model(
        mpath,
        custom_objects={
            "jaccard_index_loss": losses.JaccardIndexLoss(),
            "jaccard_index_metric": losses.JaccardIndexMetric(),
            "dice_coeff_metric": losses.DiceCoefMetric(),
            "recall": losses.recall,
            "precision": losses.precision,
            "specificity": losses.specificity,
            # F1 score is needed for old models
            # "f1_score": losses.f1_score,
            "tf": tf,
        },
    )
    model.load_weights(
        f"/media/ML/mlflow/beetle/artifacts/34/{model_hash}/"
        + "artifacts/best_weights/best_weights"
    )
    metrics, _ = evaluate_model(
        model=model,
        thr=thr,
        dpath=dpath,
        gtpath=gtpath,
        vpath=vpath,
        rpath=rpath,
        vids=vids,
        batch_size=batch_size,
        img_ids=img_ids,
        mlflow=mlflow,
        run_name=run_name,
    )
    mean_metrics = {}
    mean_metrics_snow = {}
    for key, value in metrics.items():
        mean_metrics[key] = np.mean(list(value.values()))
        mean_metrics_snow[f"snow_{key}"] = np.mean(
            [value[x] for x in snow_imgs])
    print(mean_metrics, mean_metrics_snow)
    if mlflow:
        log_metrics(mean_metrics)
        log_metrics(mean_metrics_snow)
        log_artifacts(rpath)


if __name__ == "__main__":
    clize.run(run_evaluation)
