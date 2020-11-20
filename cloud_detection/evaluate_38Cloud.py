""" Get evaluation metrics for given model on 38-Cloud testset. """

import os
import re
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from skimage import io, img_as_ubyte

import losses
from data_gen import load_image_paths, DG_38Cloud
from utils import overlay_mask, unpad, get_metrics


def true_positives(y_true, y_pred):
    return y_true * y_pred


def false_positives(y_true, y_pred):
    y_true_neg = 1 - y_true
    return y_true_neg * y_pred


def false_negatives(y_true, y_pred):
    y_pred_neg = 1 - y_pred
    return y_true * y_pred_neg


def get_full_scene_img(path: Path, img_id: str):
    img_path = next(path.glob("*" + img_id + "*"))
    return np.array(load_img(img_path))/255


def get_img_pred(path: Path, img_id: str, model: keras.Model,
                 batch_size: int, patch_size: int=384) -> np.ndarray:
    """
    Generates prediction for a given image.
    param path: path containing directories with image channels.
    param img_id: ID of the considered image.
    param model: trained model to make predictions.
    param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    param patch_size: size of the image patches.
    return: prediction for a given image.
    """
    test_files, = load_image_paths(path, [1.0], img_id)
    testgen = DG_38Cloud(
        files=test_files,
        batch_size=batch_size,
        shuffle=False,
        with_gt=False
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
        img[(row-1)*patch_size:row*patch_size,
            (col-1)*patch_size:col*patch_size] = pred[i]
    return img, scene_time


def get_img_pred_shape(files: List[Dict[str, Path]],
                       patch_size: int) -> Tuple:
    """
    Infers shape of the predictions of the considered image.
    param files: paths to patch files;
          structured as: list_of_files['file_channel', Path].
    param patch_size: size of the image patches.
    return: shape of the predictions of the considered image.
    """
    row_max, col_max = 0, 0
    for fnames in files:
        red_fname = str(fnames["red"])
        row, col = re.search("([0-9]*)_by_([0-9]*)", red_fname).groups()
        row, col = int(row), int(col)
        row_max = max(row_max, row)
        col_max = max(col_max, col)
    return (patch_size*row_max, patch_size*col_max, 1)


def load_img_gt(path: Path, fname: str) -> np.ndarray:
    """
    Load image ground truth.
    param path: path containing image gts.
    param fname: image gt file name.
    return: image ground truth.
    """
    img = np.array(load_img(path / fname, color_mode="grayscale"))
    return np.expand_dims(img/255, axis=-1)


def save_vis(img_id, vpath, img_pred, img_gt):
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")

    img_pred = np.round(img_pred)
    io.imsave("artifacts/" + img_id + "_gt.png", img_as_ubyte(img_gt[:,:,0]))
    io.imsave("artifacts/" + img_id + "_pred.png", img_as_ubyte(img_pred[:,:,0]))

    fsi = 0.7 * get_full_scene_img(vpath, img_id)
    mask_vis = overlay_mask(fsi, true_positives(img_gt, img_pred), (1, 1,  0))
    mask_vis = overlay_mask(mask_vis, false_positives(img_gt, img_pred), (1, 0, 0))
    mask_vis = overlay_mask(mask_vis, false_negatives(img_gt, img_pred), (1, 0, 1))
    io.imsave("artifacts/" + img_id + "_masks.png", img_as_ubyte(mask_vis))


def evaluate_model(model: keras.Model, dpath: Path,
                   gtpath: Path, batch_size: int) -> Tuple:
    """
    Get evaluation metrics for given model on 38-Cloud testset.
    param model: trained model to make predictions.
    param dpath: path to dataset.
    param gtpath: path to dataset ground truths.
    param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    return: evaluation metrics.
    """
    metrics = {}
    scene_times = []
    for metric_fn in model.metrics:
        if type(metric_fn) is str:
            metric_name = metric_fn
        else:
            metric_name = metric_fn.__name__
        metrics[f"38Cloud_{metric_name}"] = {}

    for fname in os.listdir(gtpath):
        img_id = fname[fname.find("LC08"):fname.find(".TIF")]
        print(f"Processing {img_id}")
        img_pred, scene_time = get_img_pred(dpath, img_id, model, batch_size)
        scene_times.append(scene_time)
        img_gt = load_img_gt(gtpath, fname)
        img_pred = unpad(img_pred, img_gt.shape)
        img_metrics = get_metrics(img_gt, img_pred, model.metrics)
        for metric_fn in model.metrics:
            if type(metric_fn) is str:
                metric_name = metric_fn
            else:
                metric_name = metric_fn.__name__
            metrics[f"38Cloud_{metric_name}"][fname] = img_metrics[f"test_{metric_name}"]
        print(f"Average inference time: { sum(scene_times) / len(scene_times) } seconds")

    return metrics


def visualise_model(model: keras.Model, dpath: Path,
                    gtpath: Path, vpath: Path, vids: Tuple[str], batch_size: int) -> Tuple:
    """
    Get evaluation metrics for given model on 38-Cloud testset.
    param model: trained model to make predictions.
    param dpath: path to dataset.
    param gtpath: path to dataset ground truths.
    param vpath: path do dataset visualisation images.
    param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    return: evaluation metrics.
    """
    for fname in os.listdir(gtpath):
        img_id = fname[fname.find("LC08"):fname.find(".TIF")]
        if img_id in vids or '*' in vids:
            print(f"Creating visualisation for {img_id}")
            img_gt = load_img_gt(gtpath, fname)
            img_pred, _ = get_img_pred(dpath, img_id, model, batch_size)
            img_pred = unpad(img_pred, img_gt.shape)
            save_vis(img_id, vpath, img_pred, img_gt)


if __name__ == "__main__":
    mpath = Path("/media/ML/mlflow/beetle/artifacts/34/4848bf5b4c464af7b6be5abb0d70f842/"
                 + "artifacts/model/data/model.h5")
    model = keras.models.load_model(
        mpath, custom_objects={
            "jaccard_index_loss": losses.Jaccard_index_loss(),
            "jaccard_index_metric": losses.Jaccard_index_metric(),
            "dice_coeff_metric": losses.Dice_coef_metric(),
            "recall": losses.recall,
            "precision": losses.precision,
            "specificity": losses.specificity,
            "f1_score": losses.f1_score
            })
    params = {
        "model": model,
        "dpath": Path("datasets/clouds/38-Cloud/38-Cloud_test"),
        "gtpath": Path("datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts"),
        "batch_size": 10
        }
    evaluate_model(**params)
