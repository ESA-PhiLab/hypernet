"""
Get evaluation metrics for given model on L8CCA dataset.

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
import uuid
import time
import clize
import numpy as np
import tensorflow as tf
from clize.parameters import multi
from einops import rearrange
from pathlib import Path
from typing import Tuple, List, Dict, Callable
from mlflow import log_metrics, log_artifacts, log_params
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

from cloud_detection import losses
from cloud_detection.data_gen import DG_L8CCA
from cloud_detection.utils import (
    unpad, get_metrics_tf, save_vis, setup_mlflow, load_l8cca_gt, make_paths
)
from cloud_detection.validate import (
    make_precision_recall,
    make_roc,
    make_activation_hist,
)


def build_rgb_scene_img(path: Path, img_id: str) -> np.ndarray:
    """Build displayable rgb image out channel slices.

    :param path: path to directory with images.
    :param img_id: id of image to be loaded.
    :return: rgb image in numpy array.
    """
    r_path = next(path.glob("*" + img_id + "_B4*"))
    g_path = next(path.glob("*" + img_id + "_B3*"))
    b_path = next(path.glob("*" + img_id + "_B2*"))
    ret = np.stack(
        [
            np.array(load_img(r_path, color_mode="grayscale")),
            np.array(load_img(g_path, color_mode="grayscale")),
            np.array(load_img(b_path, color_mode="grayscale")),
        ],
        axis=2,
    )
    return ret / ret.max()


def get_img_pred(
    path: Path, model: keras.Model, batch_size: int,
    bands: Tuple[int] = (4, 3, 2, 5),
    bands_names: Tuple[str] = ("red", "green", "blue", "nir"),
    resize: bool = False,
    normalize=True,
    standardize=False,
    patch_size: int = 384
) -> Tuple[np.ndarray, float]:
    """
    Generates prediction for a given image.

    :param path: path containing directories with image channels.
    :param model: trained model to make predictions.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param bands: band numbers to load
    :param bands_names: names of the bands to load. Should have the same number
                        of elements as bands.
    :param resize: whether to resize loaded img to gt.
    :param normalize: whether to normalize the image.
    :param standardize: whether to standardize the image.
    :param patch_size: size of the image patches.
    :return: prediction for a given image along with evaluation time.
    """
    testgen = DG_L8CCA(
        img_paths=[path], batch_size=batch_size,
        bands=bands, bands_names=bands_names, resize=resize,
        normalize=normalize, standardize=standardize, shuffle=False
    )
    tbeg = time.time()
    preds = model.predict_generator(testgen)
    scene_time = time.time() - tbeg
    print(f"Scene prediction took { scene_time } seconds")

    img_height, img_width, _ = testgen.img_shapes[0]
    preds = rearrange(
        preds,
        "(r c) dr dc b -> r c dr dc b",
        r=int(img_height / patch_size),
        c=int(img_width / patch_size),
    )
    img = np.full((img_height, img_width, 1), np.inf)
    for row in range(preds.shape[0]):
        for column in range(preds.shape[1]):
            img[
                row * patch_size: (row + 1) * patch_size,
                column * patch_size: (column + 1) * patch_size,
            ] = preds[row, column]
    return img, scene_time


def evaluate_model(
    model: keras.Model,
    thr: float,
    dpath: Path,
    rpath: Path,
    vids: Tuple[str],
    batch_size: int,
    bands: Tuple[int] = (4, 3, 2, 5),
    bands_names: Tuple[str] = ("red", "green", "blue", "nir"),
    img_ids: List[str] = None,
    resize: bool = False,
    normalize: bool = True,
    standardize: bool = False,
    metric_fns: List[Callable] = [normalized_mutual_info_score,
                                  adjusted_rand_score],
    mlflow: bool = False,
    run_name: str = None,
) -> Tuple[Dict, List]:
    """
    Get evaluation metrics for given model on L8CCA testset.

    :param model: trained model to make predictions.
    :param thr: threshold to be used during evaluation.
    :param dpath: path to dataset.
    :param rpath: path to directory where results
                  and artifacts should be logged.
    :param vids: tuple of ids of images which should be used to create
                 visualisations. If contains '*' visualisations will be
                 created for all images in the dataset.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param bands: band numbers to load
    :param bands_names: names of the bands to load. Should have the same number
                        of elements as bands.
    :param img_ids: if given, process only these images.
    :param resize: whether to resize loaded img to gt.
    :param normalize: whether to normalize the image.
    :param standardize: whether to standardize the image.
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
        metrics[f"L8CCA_{metric_name}"] = {}
    for metric_fn in metric_fns:
        metrics[f"L8CCA_{metric_fn.__name__}"] = {}

    for tname in os.listdir(dpath):
        tpath = dpath / tname
        for img_id in os.listdir(tpath):
            if img_ids is not None:
                if img_id not in img_ids:
                    continue
            print(f"Processing {tname}-{img_id}", flush=True)
            img_path = tpath / img_id
            img_pred, scene_time = get_img_pred(
                path=img_path,
                model=model,
                batch_size=batch_size,
                bands=bands,
                bands_names=bands_names,
                resize=resize,
                normalize=normalize,
                standardize=standardize
                )
            scene_times.append(scene_time)
            img_gt = load_l8cca_gt(path=img_path)
            img_pred = unpad(img_pred, img_gt.shape)
            img_metrics = get_metrics_tf(img_gt, img_pred > thr, model.metrics)
            for metric_fn in model.metrics:
                if type(metric_fn) is str:
                    metric_name = metric_fn
                else:
                    metric_name = metric_fn.__name__
                metrics[f"L8CCA_{metric_name}"][img_id] = img_metrics[
                    f"{metric_name}"
                ]
            for metric_fn in metric_fns:
                metrics[f"L8CCA_{metric_fn.__name__}"][img_id] = metric_fn(
                    img_gt.reshape(-1),
                    (img_pred > thr).reshape(-1)
                )
            print(
                "Average inference time: "
                + f"{ sum(scene_times) / len(scene_times) } seconds"
            )
            if img_id in vids or "*" in vids:
                print(f"Creating visualisation for {img_id}")
                img_vis = build_rgb_scene_img(img_path, img_id)
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
    dpath: str = "datasets/clouds/Landsat-Cloud-Cover-Assessment-"
                 + "Validation-Data-Partial",
    rpath: str = "artifacts/",
    vids: ("v", multi(min=1)),
    batch_size: int = 32,
    bands: ("b", multi(min=1)),
    bands_names: ("bn", multi(min=1)),
    img_ids: ("iid", multi(min=0)),
    resize: bool = False,
    normalize: bool = False,
    standardize: bool = False,
    mlflow: bool = False,
    run_name: str = None,
):
    """
    Load model given model hash and get evaluation metrics on L8CCA testset.

    :param model_hash: MLFlow hash of the model to load.
    :param thr: threshold to be used during evaluation.
    :param dpath: path to dataset.
    :param rpath: path to directory where results
                  and artifacts should be logged.
    :param vids: tuple of ids of images which should be used to create
                 visualisations. If contains '*' visualisations will be
                 created for all images in the dataset.
    :type vids: tuple[str]
    :param batch_size: size of generated batches, only one batch is loaded
           to memory at a time.
    :param bands: band numbers to load
    :type bands: list[int]
    :param bands_names: names of the bands to load. Should have the same number
                        of elements as bands.
    :type bands_names: list[str]
    :param img_ids: if given, process only these images.
    :type img_ids: list[int]
    :param resize: whether to resize loaded img to gt.
    :param normalize: whether to normalize the image.
    :param standardize: whether to standardize the image.
    :param mlflow: whether to use MLFlow.
    :param run_name: name of the run.
    """
    snow_imgs = ["LC82271192014287LGN00", "LC81321192014054LGN00"]
    if img_ids == []:
        img_ids = None
    else:
        snow_imgs = list(set(snow_imgs) & set(img_ids))
    dpath, rpath = make_paths(dpath, rpath)
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
        rpath=rpath,
        vids=vids,
        batch_size=batch_size,
        bands=bands,
        bands_names=bands_names,
        img_ids=img_ids,
        resize=resize,
        normalize=normalize,
        standardize=standardize,
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
