""" Get evaluation metrics for given model on 38-Cloud testset. """

import os
import re
import numpy as np
import tensorflow.keras.backend as K
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from sklearn.metrics import jaccard_score
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import losses
from data_gen import load_image_paths, DataGenerator


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
    testgen = DataGenerator(
        files=test_files,
        batch_size=batch_size,
        shuffle=False,
        with_gt=False
        )
    pred = model.predict_generator(testgen)
    img_shape = get_img_pred_shape(test_files, patch_size)
    img = np.full(img_shape, np.inf)
    for i, fnames in enumerate(test_files):
        red_fname = str(fnames["red"])
        row, col = re.search("([0-9]*)_by_([0-9]*)", red_fname).groups()
        row, col = int(row), int(col)
        img[(row-1)*patch_size:row*patch_size,
            (col-1)*patch_size:col*patch_size] = pred[i]
    return img


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


def unpad(img: np.ndarray, gt_shape: Tuple) -> np.ndarray:
    """
    Unpadding of an image to return it to its original shape.
    param img: image to unpad.
    param gt_shape: shape of the original image.
    return: unpadded image.
    """
    r, c, _ = img.shape
    r_gt, c_gt, _ = gt_shape
    r_pad = int((r-r_gt)/2)
    c_pad = int((c-c_gt)/2)
    return img[r_pad:r_pad+r_gt, c_pad:c_pad+c_gt]


def load_img_gt(path: Path, fname: str) -> np.ndarray:
    """
    Load image ground truth.
    param path: path containing image gts.
    param fname: image gt file name.
    return: image ground truth.
    """
    img = np.array(load_img(path / fname, color_mode="grayscale"))
    return np.expand_dims(img/255, axis=-1)


def get_metrics(gt: np.ndarray, pred: np.ndarray, metric_fns: List[Callable]) -> Dict:
    """
    Calculates evaluation metrics for a given image predictions.
    param gt: image ground truth.
    param pred: image predictions.
    param metric_fns: list of metric functions.
    return: evaluation metrics.
    """
    gt = K.constant(gt)
    pred = K.constant(pred)
    metrics = {}
    for metric_fn in metric_fns:
        if type(metric_fn) is str:
            metric_name = metric_fn
            metric_fn = getattr(losses, metric_fn)
        else:
            metric_name = metric_fn.__name__
        metrics[f"test_{metric_name}"] = K.eval(K.mean(metric_fn(gt, pred)))
    return metrics


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
    for metric_fn in model.metrics:
        if type(metric_fn) is str:
            metric_name = metric_fn
        else:
            metric_name = metric_fn.__name__
        metrics[f"test_{metric_name}"] = {}
    for fname in os.listdir(gtpath):
        img_id = fname[fname.find("LC08"):fname.find(".TIF")]
        print(f"Processing {img_id}")
        img_gt = load_img_gt(gtpath, fname)
        img_pred = get_img_pred(dpath, img_id, model, batch_size)
        img_pred = unpad(img_pred, img_gt.shape)
        img_metrics = get_metrics(img_gt, img_pred, model.metrics)
        for metric_fn in model.metrics:
            if type(metric_fn) is str:
                metric_name = metric_fn
            else:
                metric_name = metric_fn.__name__
            metrics[f"test_{metric_name}"][fname] = img_metrics[f"test_{metric_name}"]
    return metrics

if __name__ == "__main__":
    mpath = Path("/media/ML/mlflow/beetle/artifacts/34/f2e7e345d95e42c7b9f213f5c4af54db/"
                 + "artifacts/model/data/model.h5")
    model = keras.models.load_model(mpath,
                                    custom_objects={"loss": losses.jaccard_index()})
    params = {
        "model": model,
        "dpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test"),
        "gtpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts"),
        "batch_size": 10
        }
    evaluate_model(**params)
