"""
Various utilities, running tools, img editing etc.

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

import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = 310000000
from pathlib import Path
from scipy.stats import zscore
from tensorflow import keras
from typing import Dict, List, Tuple, Callable
from skimage import io, img_as_ubyte
from tensorflow.keras.preprocessing.image import load_img
import mlflow
import numpy as np
import spectral.io.envi as envi
import tensorflow.keras.backend as K

import cloud_detection.losses


def open_as_array(
    channel_files: Dict[str, Path],
    channel_names: Tuple[str] = ("red", "green", "blue", "nir"),
    size: Tuple[int] = None,
    normalize: bool = True,
    standardize: bool = False,
) -> np.ndarray:
    """
    Load image as array from given files. Normalises images on load.

    :param channel_files: Dict with paths to files containing each channel
                          of an image. Keys should contain channel_names.
    :param channel_names: Tuple of channel names to load.
    :param size: size of the image. If None, return original size.
    :param normalize: whether to normalize the image.
    :param standardize: whether to standardize the image
                        (separately for each band).
    :return: given image as a single numpy array.
    """
    array_img = np.stack(
        [np.array(load_img(
            channel_files[name],
            color_mode="grayscale",
            target_size=size
            )) for name in channel_names],
        axis=2,
    )
    if normalize:
        array_img = array_img / np.iinfo(array_img.dtype).max
    # Standardize should not be used for 38-Cloud dataset, because it will
    # standardize based on patches in the training, but based on images
    # in the evaluation.
    if standardize:
        array_img_shape = array_img.shape
        array_img = array_img.reshape(-1, array_img_shape[-1])
        array_img = zscore(array_img, axis=0)
        array_img = array_img.reshape(array_img_shape)
    return array_img


def load_38cloud_gt(channel_files: Dict[str, Path]) -> np.ndarray:
    """
    Load 38-Cloud ground truth mask as array from given files.

    :param channel_files: Dict with paths to files containing each channel
                            of an image, must contain key 'gt'.
    :return: patch ground truth.
    """
    masks = np.array(load_img(channel_files["gt"], color_mode="grayscale"))
    return np.expand_dims(masks / 255, axis=-1)


def load_l8cca_gt(path: Path) -> np.ndarray:
    """
    Load L8CCA Validation Data image ground truth.

    :param path: path containing image gts.
    :return: image ground truth.
    """
    img = envi.open(list(path.glob("*_fixedmask.hdr"))[0])
    img = np.array(img.open_memmap(), dtype=np.int)
    img = np.where(img > 128, 1, 0)
    return img


def true_positives(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate matrices indicating true positives in given predictions.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Array with values indicating true positives in predictions.
    """
    return y_true * y_pred


def false_positives(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate matrices indicating false positives in given predictions.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Array with values indicating false positives in predictions.
    """
    y_true_neg = 1 - y_true
    return y_true_neg * y_pred


def false_negatives(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate matrices indicating false negatives in given predictions.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Array with values indicating false negatives in predictions.
    """
    y_pred_neg = 1 - y_pred
    return y_true * y_pred_neg


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rgb_color: Tuple[float, float, float],
    overlay_intensity: float = 0.5
) -> np.ndarray:
    """
    Overlay a mask on image for visualization purposes.

    :param image: Image on which mask should be overlaid.
    :param mask: Mask which should be overlaid on the image.
    :param rgb_color: Tuple of three floats containing intensity of RGB
        channels of created mask. RBG values can be in range 0 to 1 or 0 to 255
        depending on the image and mask values range. This will effectively
        set color of the overlay mask.
    :param overlay_intensity: Intensity of the overlaid mask. Should be
        between 0 and 1.
    :return: mask overlaid on image.
    """
    image = np.copy(image)
    for i, color in enumerate(rgb_color):
        channel = image[:, :, i]
        channel += overlay_intensity * color * mask[:, :, 0]

    return np.clip(image, 0, 1)


def setup_mlflow(run_name: str):
    """
    Start mlflow run with given name.

    :param run_name: name of the run.
    """
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.start_run(run_name=run_name)


def pad(img: np.ndarray, patch_size: int = 384) -> np.ndarray:
    """
    Padding of an image to divide it into patches.

    :param img: image to pad.
    :param patch_size: size of the patches.
    :return: padded image.
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


def get_metrics_tf(
        gt: np.ndarray, pred: np.ndarray, metric_fns: List[Callable]) -> Dict:
    """
    Calculates TensorFlow evaluation metrics for a given image predictions.

    :param gt: image ground truth.
    :param pred: image predictions.
    :param metric_fns: list of metric functions.
    :return: evaluation metrics.
    """
    gt_ph = K.placeholder(ndim=4)
    pred_ph = K.placeholder(ndim=4)
    metrics = {}
    for metric_fn in metric_fns:
        if type(metric_fn) is str:
            metric_name = metric_fn
            metric_fn = getattr(cloud_detection.losses, metric_fn)
        else:
            metric_name = metric_fn.__name__
        loss = K.mean(metric_fn(gt_ph, pred_ph))
        metrics[f"{metric_name}"] = loss.eval(
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
    :param img_vis: RGB image.
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
        img_vis, true_positives(img_gt, img_pred), (1., 1., 0.))
    mask_vis = overlay_mask(
        mask_vis, false_positives(img_gt, img_pred), (1., 0., 0.))
    mask_vis = overlay_mask(
        mask_vis, false_negatives(img_gt, img_pred), (1., 0., 1.))
    io.imsave(rpath / "masks.png", img_as_ubyte(mask_vis))


def make_paths(*args: str) -> Tuple[Path]:
    """
    Make Paths out of strings.

    :params: strings to make into Paths.
    :return: Paths made out of input strings.
    """
    paths = [Path(path) if path is not None else None for path in [*args]]
    return tuple(paths)


class MLFlowCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch: int, logs: Dict = None):
        """
        Triggered after each epoch, logging metrics to MLFlow.

        :param epoch: index of epoch.
        :param logs: logs for MLFlow.
        """
        mlflow.log_metrics(logs, step=epoch)


def strip_nir(hyper_img: np.ndarray) -> np.ndarray:
    """
    Strips nir channel so image can be displayed.

    :param hyper_img: image with shape (x, y, 4) where fourth channel is nir.
    :return: image with shape (x, y, 3) with standard RGB channels.
    """
    return hyper_img[:, :, :3]


def load_image_paths(
    base_path: Path,
    patches_path: Path = None,
    split_ratios: List[float] = [1.0],
    shuffle: bool = True,
    img_id: str = None,
    seed: int = 42
) -> List[List[Dict[str, Path]]]:
    """
    Build paths to all files containing image channels.

    :param base_path: root path containing directories with image channels.
    :param patches_path: path to images patches names to load
                         (if None, all patches will be used).
    :param split_ratios: list containing split ratios,
                         splits should add up to one.
    :param shuffle: whether to shuffle image paths.
    :param img_id: image ID; if specified, load paths for this image only.
    :param seed: random seed for shuffling; relevant only if shuffle=True.
    :return: list with paths to image files, separated into splits.
        Structured as: list_of_splits[list_of_files['file_channel', Path]]
    """
    files = build_paths(base_path, patches_path, img_id)
    if len(files) == 0:
        raise ValueError("No files loaded")
    print(f"Loaded paths for images of { len(files) } samples")
    if shuffle:
        saved_seed = np.random.get_state()
        np.random.seed(seed)
        np.random.shuffle(files)
        np.random.set_state(saved_seed)
    if sum(split_ratios) != 1:
        raise RuntimeError("Split ratios don't sum up to one.")
    split_beg = 0
    splits = []
    for ratio in split_ratios:
        split_end = split_beg + math.ceil(ratio * len(files))
        splits.append(files[split_beg:split_end])
        split_beg = split_end
    return splits


def combine_channel_files(red_file: Path) -> Dict[str, Path]:
    """
    Get paths to 'green', 'blue', 'nir' and 'gt' channel files
    based on path to the 'red' channel of the given image.

    :param red_file: path to red channel file.
    :return: dictionary containing paths to files with each image channel.
    """
    return {
        "red": red_file,
        "green": Path(str(red_file).replace("red", "green")),
        "blue": Path(str(red_file).replace("red", "blue")),
        "nir": Path(str(red_file).replace("red", "nir")),
        "gt": Path(str(red_file).replace("red", "gt")),
    }


def build_paths(
    base_path: Path, patches_path: Path, img_id: str
) -> List[Dict[str, Path]]:
    """
    Build paths to all files containing image channels.

    :param base_path: root path containing directories with image channels.
    :param patches_path: path to images patches names to load
                            (if None, all patches will be used).
    :param img_id: image ID; if specified, load paths for this image only.
    :return: list of dicts containing paths to files with image channels.
    """
    # Get red channel filenames
    if img_id is None:
        red_files = list(base_path.glob("*red/*.TIF"))
    else:
        red_files = list(base_path.glob(f"*red/*{img_id}.TIF"))
    if patches_path is not None:
        patches_names = set(
            np.genfromtxt(
                patches_path,
                dtype="str",
                skip_header=1,
            )
        )
        select_files = []
        for fname in red_files:
            fname_str = str(fname)
            if (
                fname_str[fname_str.find("patch"): fname_str.find(".TIF")]
                in patches_names
            ):
                select_files.append(fname)
        red_files = select_files
    red_files.sort()
    # Get other channels in accordance to the red channel filenames
    return [combine_channel_files(red_file) for red_file in red_files]
