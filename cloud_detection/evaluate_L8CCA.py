""" Get evaluation metrics for given model on L8CCA dataset. """

import os
import uuid
import time
import argparse
import numpy as np
import spectral.io.envi as envi
import tensorflow as tf
from einops import rearrange
from pathlib import Path
from typing import Tuple, List
from mlflow import log_metrics, log_artifacts
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import losses
from data_gen import DG_L8CCA
from utils import unpad, get_metrics, save_vis
from validate import make_precission_recall, make_roc, make_activation_hist
from validate import datagen_to_gt_array


def build_rgb_scene_img(path: Path, img_id: str) -> np.ndarray:
    """ Build displayable rgb image out channel slices.
    :param path: path to directory with images.
    :param img_id: id of image to be loaded.
    :return: rgb image in numpy array.
    """
    r_path = next(path.glob("*" + img_id + "_B4*"))
    g_path = next(path.glob("*" + img_id + "_B3*"))
    b_path = next(path.glob("*" + img_id + "_B2*"))
    ret = np.stack([
        np.array(load_img(r_path, color_mode="grayscale")),
        np.array(load_img(g_path, color_mode="grayscale")),
        np.array(load_img(b_path, color_mode="grayscale")),
    ], axis=2)
    return (ret / ret.max())
    

def get_img_pred(path: Path, img_id: str, model: keras.Model,
                 batch_size: int, patch_size: int = 384) -> np.ndarray:
    """
    Generates prediction for a given image.
    :param path: path containing directories with image channels.
    :param img_id: ID of the considered image.
    :param model: trained model to make predictions.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param patch_size: size of the image patches.
    :return: prediction for a given image.
    """
    testgen = DG_L8CCA(
        img_path=path,
        img_name=img_id,
        batch_size=batch_size,
        shuffle=False
        )
    tbeg = time.time()
    preds = model.predict_generator(testgen)
    scene_time = time.time() - tbeg
    print(f"Scene prediction took { scene_time } seconds")

    img_shape = testgen.img_shape
    preds = rearrange(preds, '(r c) dr dc b -> r c dr dc b',
                      r=int(img_shape[0]/patch_size),
                      c=int(img_shape[1]/patch_size))
    img = np.full((img_shape[0], img_shape[1], 1), np.inf)
    for r in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            img[r*patch_size:(r+1)*patch_size,
                c*patch_size:(c+1)*patch_size] = preds[r, c]
    return img, scene_time


def load_img_gt(path: Path, fname: str) -> np.ndarray:
    """
    Load image ground truth.
    :param path: path containing image gts.
    :param fname: image gt file name.
    :return: image ground truth.
    """
    img = envi.open(path / fname)
    img = np.asarray(img[:, :, :], dtype=np.int)
    img = np.where(img > 128, 1, 0)
    return img


def evaluate_model(model: keras.Model, thr: float, dpath: Path,
                   rpath: Path, vids: Tuple[str],
                   batch_size: int, img_ids: List[str]=None,
                   mlflow=False, run_name=None) -> Tuple:
    """
    Get evaluation metrics for given model on 38-Cloud testset.
    :param model: trained model to make predictions.
    :param thr: threshold.
    :param dpath: path to dataset.
    :param rpath: path to direcotry where results and artifacts should be logged.
    :param vids: tuple of ids of images which should be used to create visualisations.
        If contains '*' visualisations will be created for all images in the dataset.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param img_ids: if given, process only these images.
    :return: evaluation metrics.
    """
    Path(rpath).mkdir(parents=True, exist_ok=False)
    if mlflow == True:
        setup_mlflow(locals())
    metrics = {}
    scene_times = []
    for metric_fn in model.metrics:
        if type(metric_fn) is str:
            metric_name = metric_fn
        else:
            metric_name = metric_fn.__name__
        metrics[f"L8CCA_{metric_name}"] = {}

    for tname in os.listdir(dpath):
        tpath = dpath / tname
        for img_id in os.listdir(tpath):
            if img_ids is not None:
                if img_id not in img_ids:
                    continue
            print(f"Processing {tname}-{img_id}", flush=True)
            gtpath = tpath / img_id
            img_pred, scene_time = get_img_pred(tpath,
                                                img_id,
                                                model,
                                                batch_size)
            scene_times.append(scene_time)
            img_gt = load_img_gt(gtpath, f"{img_id}_fixedmask.hdr")
            img_pred = unpad(img_pred, img_gt.shape)
            img_metrics = get_metrics(img_gt, img_pred > thr, model.metrics)
            for metric_fn in model.metrics:
                if type(metric_fn) is str:
                    metric_name = metric_fn
                else:
                    metric_name = metric_fn.__name__
                metrics[f"L8CCA_{metric_name}"][img_id] = \
                    img_metrics[f"test_{metric_name}"]
            print("Average inference time: "
                  + f"{ sum(scene_times) / len(scene_times) } seconds")
            if img_id in vids or '*' in vids:
                print(f"Creating visualisation for {img_id}")
                img_vis = build_rgb_scene_img(tpath/img_id, img_id)
                save_vis(img_id, img_vis, img_pred, img_gt, rpath)

            if img_metrics['test_jaccard_index_metric'] < 0.6:
                print(f"Will make insights for {img_id}", flush=True)
                y_gt = img_gt.ravel()
                y_pred = np.round(img_pred.ravel(), decimals=5)

                make_roc(y_gt, y_pred, rpath / img_id, thr_marker=thr)
                make_precission_recall(y_gt, y_pred, rpath / img_id, thr_marker=thr)

                # Make histogram with more rounded predictions for performance reasons
                y_pred = np.round(y_pred, decimals=2)
                make_activation_hist(y_pred, rpath / img_id)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)

    args = parser.parse_args()

    mpath = Path("/media/ML/mlflow/beetle/artifacts/34/3e19daf248954674966cd31af1c4cb12/"
                 + "artifacts/model/data/model.h5")
    model = keras.models.load_model(
        mpath, custom_objects={
            "jaccard_index_loss": losses.Jaccard_index_loss(),
            "jaccard_index_metric": losses.Jaccard_index_metric(),
            "dice_coeff_metric": losses.Dice_coef_metric(),
            "recall": losses.recall,
            "precision": losses.precision,
            "specificity": losses.specificity,
            "f1_score": losses.f1_score,
            "tf": tf
            })
    # TODO: fall back from 1 to .5
    params = {
        "model": model,
        "thr": 0.5,
        "dpath": Path("../datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial"),
        "rpath": Path(f"artifacts/{uuid.uuid4().hex}"),
        "vids": ("*"),
        "batch_size": 32,
        "mlflow": args.f,
        "run_name": args.n
        }
    metrics = evaluate_model(**params)
    snow_imgs = ["LC82271192014287LGN00",
                 "LC81321192014054LGN00"]
    mean_metrics = {}
    mean_metrics_snow = {}
    for key, value in metrics.items():
        mean_metrics[key] = np.mean(list(value.values()))
        mean_metrics_snow[f"snow_{key}"] = np.mean([value[x] for x in snow_imgs])
    print(mean_metrics, mean_metrics_snow)
    if params["mlflow"] == True:
        log_metrics(mean_metrics)
        log_metrics(mean_metrics_snow)
        log_artifacts(params["rpath"])
