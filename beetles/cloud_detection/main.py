""" Train and test the models for cloud detection. """

import uuid
import os
import yaml
import numpy as np
import argparse
from mlflow import log_metrics, log_artifacts, log_param, log_params, end_run
from typing import Tuple, List

from cloud_detection.train_model import train_model
from cloud_detection.evaluate_38Cloud import evaluate_model as test_38Cloud
from cloud_detection.evaluate_L8CCA import evaluate_model as test_L8CCA
from cloud_detection.utils import setup_mlflow, make_paths


def main(
    run_name: str,
    train_path: str,
    C38_path: str,
    C38_gtpath: str,
    L8CCA_path: str,
    vpath: str,
    rpath: str,
    ppath: str,
    vids: Tuple[str],
    mlflow: bool,
    train_size: float,
    train_img: str,
    balance_train_dataset: bool,
    balance_val_dataset: bool,
    balance_snow: bool,
    snow_imgs_38Cloud: List[str],
    snow_imgs_L8CCA: List[str],
    batch_size: int,
    thr: float,
    learning_rate: float,
    bn_momentum: float,
    epochs: int,
    stopping_patience: int,
):
    """
    Train and test the U-Net model using 38-Cloud and L8CCA datasets.

    :param run_name: name of the run.
    :param train_path: path to train dataset.
    :param C38_path: path to 38-Cloud dataset.
    :param C38_gtpath: path to 38-Cloud groundtruth.
    :param L8CCA_path: path to L8CCA dataset.
    :param vpath: path to 38-Cloud dataset (false color) visualisation images.
    :param rpath: path to directory where results and artifacts should be
                  logged (randomly named directory will be created to store the
                  results).
    :param ppath: path to file with names of training patches
                  (if None, all training patches will be used).
    :param vids: tuple of ids of images which should be used to create
                 visualisations. If contains '*' visualisations will be
                 created for all images in the datasets.
    :param mlflow: whether to use mlflow
    :param train_size: proportion of the training set
                       (the rest goes to validation set).
    :param train_img: image ID for training; if specified,
                      load training patches for this image only.
    :param balance_train_dataset: whether to balance train dataset.
    :param balance_val_dataset: whether to balance val dataset.
    :param balance_snow: whether to balance snow images.
    :param snow_imgs_38Cloud: list of 38-Cloud snow images IDs for testing.
    :param snow_imgs_L8CCA: list of L8CCA snow images IDs for testing.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param thr: threshold for determining whether pixels contain the clouds
                (if None, threshold will be determined automatically).
    :param learning_rate: learning rate for training.
    :param bn_momentum: momentum of the batch normalization layer.
    :param epochs: number of epochs.
    :param stopping_patience: patience param for early stopping.
    """
    train_path, C38_path, C38_gtpath, L8CCA_path, vpath, rpath, ppath = \
        make_paths(
            train_path, C38_path, C38_gtpath, L8CCA_path, vpath, rpath, ppath
        )
    rpath = rpath / uuid.uuid4().hex
    rpath.mkdir(parents=True, exist_ok=False)
    if mlflow:
        setup_mlflow(run_name)
        log_params(locals())
    model, auto_thr = train_model(
        train_path,
        rpath,
        ppath,
        train_size,
        batch_size,
        balance_train_dataset,
        balance_val_dataset,
        balance_snow,
        train_img,
        bn_momentum,
        learning_rate,
        stopping_patience,
        epochs,
        mlflow,
    )
    print("Finished training and validation, starting evaluation.", flush=True)
    print(f"Working dir: {os.getcwd()}, artifacts dir: {rpath}", flush=True)
    thr = auto_thr if thr is None else thr
    metrics_38Cloud = test_38Cloud(
        model, thr, C38_path, C38_gtpath, vpath,
        rpath / "38Cloud_vis", vids, batch_size
    )
    mean_metrics_38Cloud = {}
    mean_metrics_38Cloud_snow = {}

    for key, value in metrics_38Cloud.items():
        mean_metrics_38Cloud[key] = np.mean(list(value.values()))
        mean_metrics_38Cloud_snow[f"snow_{key}"] = np.mean(
            [value[x] for x in snow_imgs_38Cloud]
        )

    if mlflow:
        log_param("threshold", thr)
        log_metrics(mean_metrics_38Cloud)
        log_metrics(mean_metrics_38Cloud_snow)

    metrics_L8CCA = test_L8CCA(
        model, thr, L8CCA_path, rpath / "L8CCA_vis", vids, batch_size
    )
    mean_metrics_L8CCA = {}
    mean_metrics_L8CCA_snow = {}

    for key, value in metrics_L8CCA.items():
        mean_metrics_L8CCA[key] = np.mean(list(value.values()))
        mean_metrics_L8CCA_snow[f"snow_{key}"] = np.mean(
            [value[x] for x in snow_imgs_L8CCA]
        )

    if mlflow:
        log_metrics(mean_metrics_L8CCA)
        log_metrics(mean_metrics_L8CCA_snow)
        log_artifacts(rpath)
        end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)
    parser.add_argument(
        "-c", help="config path", default="cloud_detection/cfg/exp_1.yml"
    )
    args = parser.parse_args()

    with open(args.c, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["exp_cfg"]["run_name"] = args.n
    cfg["exp_cfg"]["mlflow"] = args.f

    main(**cfg["exp_cfg"], **cfg["train_cfg"])
