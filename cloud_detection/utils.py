from typing import Dict, List, Tuple, Callable

import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow.keras.backend as K

import losses


def overlay_mask(image: np.ndarray, mask: np.ndarray, rgb_color: Tuple[float], overlay_intensity: float=0.5) -> np.ndarray:
    """ Overlay a mask on image for visualization purposes. """
    for i, color in enumerate(rgb_color):
        channel = image[:,:,i]
        channel += overlay_intensity * color *  mask[:,:,0]

    return np.clip(image, 0, 1)


def setup_mlflow(c):
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.tensorflow.autolog(every_n_iter=1)
    mlflow.log_params(c)


def pad(img: np.ndarray, patch_size: int = 384) -> np.ndarray:
    """
    Padding of an image to divide it into patches.
    param img: image to pad.
    param patch_size: size of the patches.
    return: unpadded image.
    """
    x_len, y_len, _ = img.shape
    x_r = (-x_len) % patch_size
    y_r = (-y_len) % patch_size
    x_l_pad, x_r_pad = int(np.floor(x_r/2)), int(np.ceil(x_r/2))
    y_l_pad, y_r_pad = int(np.floor(y_r/2)), int(np.ceil(y_r/2))
    return np.pad(img, ((x_l_pad, x_r_pad), (y_l_pad, y_r_pad), (0, 0)))


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
