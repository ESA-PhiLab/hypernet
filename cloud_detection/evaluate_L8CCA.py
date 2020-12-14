""" Get evaluation metrics for given model on L8CCA dataset. """

import os
import uuid
import time
import numpy as np
import spectral.io.envi as envi
from einops import rearrange
from pathlib import Path
from typing import Tuple
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

import losses
from data_gen import DG_L8CCA
from utils import unpad, get_metrics, save_vis


def build_rgb_scene_img(path: Path, img_id: str):
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
    param path: path containing directories with image channels.
    param img_id: ID of the considered image.
    param model: trained model to make predictions.
    param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    param patch_size: size of the image patches.
    return: prediction for a given image.
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
    param path: path containing image gts.
    param fname: image gt file name.
    return: image ground truth.
    """
    img = envi.open(path / fname)
    img = np.asarray(img[:, :, :], dtype=np.int)
    img = np.where(img > 128, 1, 0)
    return img


def evaluate_model(model: keras.Model, dpath: Path,
                   rpath: Path, vids: Tuple[str],
                   batch_size: int) -> Tuple:
    """
    Get evaluation metrics for given model on 38-Cloud testset.
    param model: trained model to make predictions.
    param dpath: path to dataset.
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
        metrics[f"L8CCA_{metric_name}"] = {}

    for tname in os.listdir(dpath):
        tpath = dpath / tname
        for img_id in os.listdir(tpath):
            print(f"Processing {tname}-{img_id}")
            gtpath = tpath / img_id
            img_pred, scene_time = get_img_pred(tpath,
                                                img_id,
                                                model,
                                                batch_size)
            scene_times.append(scene_time)
            img_gt = load_img_gt(gtpath, f"{img_id}_fixedmask.hdr")
            img_pred = unpad(img_pred, img_gt.shape)
            img_metrics = get_metrics(img_gt, img_pred, model.metrics)
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
            

    return metrics


if __name__ == "__main__":
    mpath = Path("/media/ML/mlflow/beetle/artifacts/34/f7129e6a404443f591d2e2ad6860c04b/"
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
        "dpath": Path("datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial"),
        "rpath": Path(f"cloud_detection/artifacts/{uuid.uuid4().hex}"),
        "vids": ("*"),
        "batch_size": 10
        }
    evaluate_model(**params)
