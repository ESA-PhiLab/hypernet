"""
Perform various validation tasks, like making ROC,
precision-recall, thresholding etc.
"""

from collections.abc import Sequence
from pathlib import Path

from plotly import express as px
from plotly import graph_objects as go
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow import keras
import numpy as np
import tensorflow as tf

from cloud_detection.data_gen import load_image_paths, DG_38Cloud
import cloud_detection.losses


def datagen_to_gt_array(datagen: keras.utils.Sequence) -> np.ndarray:
    """
    Load all gt masks from data generator and store them in RAM at once
    inside a np.array.
    :param datagen: data generator to load gt masks from.
    :return: array of gt masks.
    """
    ret = []
    # Keras internals force usage of range(len()) here instead of enumerate
    for i in range(len(datagen)):
        ret.append(datagen[i][1])

    return np.concatenate(ret)


def find_best_thr(fpr: np.ndarray, tpr: np.ndarray, thr: np.ndarray) -> float:
    """
    Find best threshold based on ROC curve.
    :param fpr: False positives rate ROC axis.
    :param tpr: True positives rate ROC axis.
    :param thr: Thresholds corresponding to ROC points.
    :return: Best thr based on minimal distance to point (0, 1) on ROC.
    """

    curve_points = np.transpose(np.vstack((fpr, tpr)))
    perfect_point = [(0.0, 1.0)]
    dists = cdist(curve_points, perfect_point, "euclidean")
    print("thr dist variance:", np.var(dists[1:-1]))
    print("thr dist mean:", np.mean(dists[1:-1]))
    best_idx = np.argmin(dists)
    return thr[best_idx]


def make_roc(
    y_gt: np.ndarray, y_pred: np.ndarray,
    output_dir: Path, thr_marker: float = None
) -> float:
    """
    Make ROC curve and save it, get best thr based on the ROC.
    :param y_gt: Array of ground truth masks.
    :param y_pred: Array of predictions.
    :param output_dir: Path to directory where ROC should be saved.
    :param thr_marker: extra threshold level to mark on ROC.
    :return: Best thr based on minimal distance to point (0, 1) on ROC.
    """
    fpr, tpr, thr = roc_curve(y_gt, y_pred)

    best_thr = find_best_thr(fpr, tpr, thr)
    print("Optimal thr:", best_thr)

    fig = px.area(
        x=fpr,
        y=tpr,
        hover_data={"threshold": thr},
        title=f"ROC Curve (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    if thr_marker is not None:
        marker_idx = find_nearest(thr, thr_marker)
        fig.add_trace(
            go.Scatter(
                x=[fpr[marker_idx]],
                y=[tpr[marker_idx]],
                text=f"set threshold: {thr_marker}",
            )
        )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fig.layout.update(showlegend=False)
    fig.write_html(str(output_dir / "roc.html"))
    return best_thr


def make_activation_hist(y_pred: np.ndarray, output_dir: Path):
    """
    Make histogram of prediction activations.
    :param y_pred: Array of predictions.
    :param output_dir: Path to directory where hist should be saved.
    """
    fig = px.histogram(y_pred.ravel(), log_y=True)
    fig.layout.update(showlegend=False)
    fig.write_html(str(output_dir / "activation_hist.html"))


def make_precision_recall(
    y_gt: np.ndarray, y_pred: np.ndarray,
    output_dir: Path, thr_marker: float = None
):
    """
    Make precision recall curve and save it.
    :param y_gt: Array of ground truth masks.
    :param y_pred: Array of predictions.
    :param output_dir: Path to directory where curve should be saved.
    :param thr_marker: Extra threshold level to mark on the curve.
    """
    fpr, fre, thr = precision_recall_curve(y_gt, y_pred)

    fig = px.area(
        x=fre,
        y=fpr,
        hover_data={"threshold": np.insert(thr, 0, 1)},
        title="Precision-Recall Curve",
        labels=dict(x="Recall", y="Precision"),
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=1, y1=0)
    if thr_marker is not None:
        marker_idx = find_nearest(thr, thr_marker)
        fig.add_trace(
            go.Scatter(
                x=[fpr[marker_idx]],
                y=[fre[marker_idx]],
                text=f"set threshold: {thr_marker}",
            )
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fig.layout.update(showlegend=False)
    fig.write_html(str(output_dir / "prec_recall.html"))


def make_validation_insights(
    model: keras.Model, datagen: keras.utils.Sequence, output_dir: Path
) -> float:
    """
    Make validation insights including:
        * Making ROC and finding best thr on ROC.
        * Making precision-recall curve.
    :param model: Model to validate.
    :param datagen: Validation data generator.
    :param output_dir: Path to directory where artifacts should be saved.
    :return: Best thr based on ROC.
    """
    output_dir.mkdir(exist_ok=True)

    y_gt = datagen_to_gt_array(datagen).ravel()
    y_pred = np.round(model.predict_generator(datagen).ravel(), decimals=3)
    print("Loaded whole validation dataset to memory", flush=True)
    best_thr = make_roc(y_gt, y_pred, output_dir)
    make_precision_recall(y_gt, y_pred, output_dir)
    return best_thr


def find_nearest(array: Sequence, value: float) -> float:
    """
    In an array find a value that is closest to the given one.
    :param array: Sequence inside which closest value will be found.
    :param value: Value for which the closest one in the array will be found.
    :return: value inside the 'array' param that is closest to the 'value'
        param.
    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def validate_demo():
    """ Demo of the validation function. """
    train_size = 0.8
    dpath = Path("../datasets/clouds/38-Cloud/38-Cloud_training")
    batch_size = 8

    _, val_files = load_image_paths(
        base_path=dpath,
        split_ratios=(train_size, 1 - train_size),
    )
    valgen = DG_38Cloud(
        files=val_files,
        batch_size=batch_size,
    )

    mpath = Path(
        "/media/ML/mlflow/beetle/artifacts/34/"
        + "b943e6e7066a458f8037b63dc1a960a3/artifacts/model/data/model.h5"
    )
    model = keras.models.load_model(
        mpath,
        custom_objects={
            "jaccard_index_loss": cloud_detection.losses.JaccardIndexLoss(),
            "jaccard_index_metric":
            cloud_detection.losses.JaccardIndexMetric(),
            "dice_coeff_metric": cloud_detection.losses.DiceCoefMetric(),
            "recall": cloud_detection.losses.recall,
            "precision": cloud_detection.losses.precision,
            "specificity": cloud_detection.losses.specificity,
            "f1_score": cloud_detection.losses.f1_score,
            "tf": tf,
        },
    )
    make_validation_insights(model, valgen, Path("./artifacts"))


if __name__ == "__main__":
    validate_demo()
