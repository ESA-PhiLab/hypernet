""" Various utilities, running tools, img editing etc. """

from pathlib import Path
from tensorflow import keras
from typing import Dict, List, Tuple, Callable

from skimage import io, img_as_ubyte
import mlflow
import numpy as np
import tensorflow.keras.backend as K

import cloud_detection.losses


def true_positives(y_true: np.ndarray, y_pred: np.ndarray):
    return y_true * y_pred


def false_positives(y_true: np.ndarray, y_pred: np.ndarray):
    y_true_neg = 1 - y_true
    return y_true_neg * y_pred


def false_negatives(y_true: np.ndarray, y_pred: np.ndarray):
    y_pred_neg = 1 - y_pred
    return y_true * y_pred_neg


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rgb_color: Tuple[float],
    overlay_intensity: float = 0.5,
) -> np.ndarray:
    """Overlay a mask on image for visualization purposes."""
    image = np.copy(image)
    for i, color in enumerate(rgb_color):
        channel = image[:, :, i]
        channel += overlay_intensity * color * mask[:, :, 0]

    return np.clip(image, 0, 1)


def setup_mlflow(run_name: str):
    """Start mlflow run with given name."""
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.start_run(run_name=run_name)


def pad(img: np.ndarray, patch_size: int = 384) -> np.ndarray:
    """
    Padding of an image to divide it into patches.
    param img: image to pad.
    param patch_size: size of the patches.
    return: padded image.
    """
    x_len, y_len, _ = img.shape
    x_r = (-x_len) % patch_size
    y_r = (-y_len) % patch_size
    x_l_pad, x_r_pad = int(np.floor(x_r / 2)), int(np.ceil(x_r / 2))
    y_l_pad, y_r_pad = int(np.floor(y_r / 2)), int(np.ceil(y_r / 2))
    return np.pad(img, ((x_l_pad, x_r_pad), (y_l_pad, y_r_pad), (0, 0)))


def unpad(img: np.ndarray, gt_shape: Tuple) -> np.ndarray:
    """
    Unpadding of an image to return it to its original shape.
    :param img: image to unpad.
    :param gt_shape: shape of the original image.
    :return: unpadded image.
    """
    r, c, _ = img.shape
    r_gt, c_gt, _ = gt_shape
    r_pad = int((r - r_gt) / 2)
    c_pad = int((c - c_gt) / 2)
    return img[r_pad: r_pad + r_gt, c_pad: c_pad + c_gt]


def get_metrics(
        gt: np.ndarray, pred: np.ndarray, metric_fns: List[Callable]) -> Dict:
    """
    Calculates evaluation metrics for a given image predictions.
    :param gt: image ground truth.
    :param pred: image predictions.
    :param metric_fns: list of metric functions.
    :return: evaluation metrics.
    """
    gt_ph = K.placeholder(ndim=3)
    pred_ph = K.placeholder(ndim=3)
    metrics = {}
    for metric_fn in metric_fns:
        if type(metric_fn) is str:
            metric_name = metric_fn
            metric_fn = getattr(cloud_detection.losses, metric_fn)
        else:
            metric_name = metric_fn.__name__
        loss = K.mean(metric_fn(gt_ph, pred_ph))
        metrics[f"test_{metric_name}"] = loss.eval(
            session=K.get_session(), feed_dict={gt_ph: gt, pred_ph: pred}
        )
    return metrics


def save_vis(
    img_id: str,
    img_vis: np.ndarray,
    img_pred: np.ndarray,
    img_gt: np.ndarray,
    rpath: Path,
):
    """
    Save visualisations set for img of given id.
    Visualisations set includes:
        * Mask overlay of uncertain regions of segmentation.
        * Ground truth mask.
        * Prediction mask.
        * TP, FP, FN mask overlays.
    :param img_id: Id of visualised img,
                   will be used for naming saved artifacts.
    :param img_pred: Prediction mask, result of segmentation.
    :param img_gt: Ground truth mask.
    :param rpath: Path where artifacts should be saved.
    """
    rpath = rpath / img_id
    Path(rpath).mkdir(parents=True, exist_ok=False)

    unc = np.copy(img_pred)
    unc[unc < 0.001] = 0
    unc[unc > 0.999] = 0
    unc[unc != 0] = 1
    unc_vis = overlay_mask(img_vis, unc, (1, 1, 0), 1.0)
    io.imsave(rpath / "unc.png", img_as_ubyte(unc_vis))

    img_pred = np.round(img_pred)
    io.imsave(rpath / "gt.png", img_gt[:, :, 0])
    io.imsave(rpath / "pred.png", img_as_ubyte(img_pred[:, :, 0]))

    mask_vis = overlay_mask(
        img_vis, true_positives(img_gt, img_pred), (1, 1, 0))
    mask_vis = overlay_mask(
        mask_vis, false_positives(img_gt, img_pred), (1, 0, 0))
    mask_vis = overlay_mask(
        mask_vis, false_negatives(img_gt, img_pred), (1, 0, 1))
    io.imsave(rpath / "masks.png", img_as_ubyte(mask_vis))


def make_paths(*args):
    paths = [Path(path) if path is not None else None for path in [*args]]
    return tuple(paths)


class MLFlowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics(logs, step=epoch)
