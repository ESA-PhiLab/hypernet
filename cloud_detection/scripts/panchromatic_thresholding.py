"""
Perform panchromatic thresholding.
"""

import os
import uuid
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from pathlib import Path
from collections import defaultdict
from mlflow import log_artifacts, log_params, end_run, log_metrics, set_tags
from tensorflow.keras.metrics import binary_crossentropy, binary_accuracy

from cloud_detection import losses
from cloud_detection.utils import (
    open_as_array, load_l8cca_gt, get_metrics, setup_mlflow
)


class ThresholdingClassifier():
    def __init__(self, thr_prop: float = 0.1):
        """
        :param thr_prop: threshold to classify pixels into clouds,
                         e.g. for thr_prop = 0.1, min pixel value = 0,
                         max pixel value = 5, all pixels with value
                         greater or equal to 0.5 will be classified as clouds.
        """
        self.thr_prop = thr_prop

    def fit(self, X: np.ndarray):
        """
        Fit method.
        :param X: image for fitting.
        :return: self.
        """
        self.min = np.min(X[X != 0])
        self.max = np.max(X[X != 0])
        self.thr = self.min + (self.max - self.min)*self.thr_prop
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict method.
        :param X: image for predicting.
        :return: cloud mask of the same shape as X,
                 where 0-not cloud & 1-cloud.
        """
        return (X >= self.thr).astype(int)


def run_panchromatic_thresholding(
    thresholds: np.ndarray = np.arange(0., 1.05, 0.05),
    dpath: Path = Path("datasets/clouds/Landsat-Cloud-Cover-Assessment-"
                       + "Validation-Data-Partial"),
    rpath: Path = Path("artifacts/"),
    band_num: int = 8,
    img_ids: List[List[str]] = [["Urban", "LC80460282014171LGN00"],
                                ["Barren", "LC81390292014135LGN00"],
                                ["Forest", "LC80160502014041LGN00"],
                                ["Grass-crops", "LC81220422014096LGN00"],
                                ["Shrubland", "LC80750172013163LGN00"],
                                ["Snow-ice", "LC81321192014054LGN00"],
                                ["Water", "LC82150712013152LGN00"],
                                ["Wetlands", "LC81010142014189LGN00"],
                                ["Barren", "LC81640502013179LGN01"],
                                ["Forest", "LC81720192013331LGN00"],
                                ["Grass-crops", "LC82020522013141LGN01"],
                                ["Shrubland", "LC80670172014206LGN00"],
                                ["Snow-ice", "LC82271192014287LGN00"],
                                ["Urban", "LC81940222013245LGN00"],
                                ["Water", "LC81910182013240LGN00"],
                                ["Wetlands", "LC81500152013225LGN00"]],
    metric_fns: List = [losses.JaccardIndexLoss(),
                        losses.JaccardIndexMetric(),
                        losses.DiceCoefMetric(),
                        losses.recall,
                        losses.precision,
                        losses.specificity,
                        binary_crossentropy,
                        binary_accuracy],
    mlflow: bool = False,
    run_name: str = None
):
    """
    Perform panchromatic thresholding for given threshold values.

    :param thresholds: threshold values to perform panchromatic thresholding.
    :param dpath: path to dataset.
    :param rpath: path to directory where results and artifacts should be
                  logged (randomly named directory will be created to store the
                  results).
    :param band_num: number of the panchromatic band.
    :param img_ids: images to process, in the form [img type, img ID].
    :param metric_fns: metric functions to run evaluation of the thresholding.
    :param mlflow: whether to use mlflow.
    :param run_name: name of the run.
    """
    rpath = rpath / uuid.uuid4().hex
    rpath.mkdir(parents=True, exist_ok=False)
    print(f"Working dir: {os.getcwd()}, artifacts dir: {rpath}", flush=True)
    if mlflow:
        setup_mlflow(run_name)
        params = dict(locals())
        del params["img_ids"]
        del params["metric_fns"]
        log_params(params)
        set_tags({"img_ids": img_ids,
                  "metric_fns": metric_fns})
    img_paths = [dpath / id_[0] / id_[1] for id_ in img_ids]
    for thr in thresholds:
        print("THRESHOLD:", thr, flush=True)
        (rpath / f"thr_{int(thr*100)}").mkdir(exist_ok=False)
        metrics_aggr = defaultdict(list)
        for i, img_path in enumerate(img_paths):
            img_type, img_name = img_path.parent.name, img_path.name
            print(img_type, img_name, flush=True)
            # Load gt
            gt = load_l8cca_gt(img_path)
            # Load img
            channel_files = {}
            channel_files["panchromatic"] = list(
                img_path.glob(f"*_B{band_num}.TIF"))[0]
            img = open_as_array(
                channel_files=channel_files,
                channel_names=("panchromatic",),
                size=gt.shape,
                normalize=False,
                standardize=False,
                )
            # Create & fit classifier
            thr_classifier = ThresholdingClassifier(thr_prop=thr).fit(img)
            # Get cloud mask & print metrics & make visualisations
            mask = thr_classifier.predict(img)
            metrics = get_metrics(gt, mask, metric_fns)
            print(metrics, flush=True)
            for k, v in metrics.items():
                metrics_aggr[k].append(v)
            fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
            fig.suptitle(f"{img_type}-{img_name}")
            axs[0].imshow(gt[:, :, 0])
            axs[0].set_title("GT")
            axs[1].imshow(mask[:, :, 0])
            axs[1].set_title("pred")
            fig.savefig(
                rpath / f"thr_{int(thr*100)}" / f"{img_type}-{img_name}.png")
            plt.close(fig)
        metrics_mean = {}
        for k, v in metrics_aggr.items():
            metrics_mean[k] = np.mean(v)
        if mlflow:
            log_metrics(metrics_mean, step=int(thr*100))
            log_artifacts(rpath)
    if mlflow:
        end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)
    args = parser.parse_args()
    run_panchromatic_thresholding(mlflow=args.f, run_name=args.n)
