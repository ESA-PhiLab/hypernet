""" Train and test the models for cloud detection. """

import uuid
import os
import yaml
import numpy as np
import argparse
from pathlib import Path
from mlflow import log_metrics, log_artifacts, log_param, log_params, end_run

from cloud_detection.train_model import train_model
from cloud_detection.evaluate_38Cloud import evaluate_model as test_38Cloud
from cloud_detection.evaluate_L8CCA import evaluate_model as test_L8CCA
from cloud_detection.utils import setup_mlflow, make_paths


def main(run_name, train_path, C38_path, C38_gtpath, L8CCA_path, vpath, rpath, ppath, vids, mlflow, train_size, train_img,
         balance_train_dataset, balance_val_dataset, balance_snow, snow_imgs_38Cloud, snow_imgs_L8CCA, batch_size, thr,
         learning_rate, bn_momentum, epochs, stopping_patience):
    train_path, C38_path, C38_gtpath, L8CCA_path, vpath, rpath, ppath = make_paths(
        train_path, C38_path, C38_gtpath, L8CCA_path, vpath, rpath, ppath)
    rpath = rpath / uuid.uuid4().hex
    rpath.mkdir(parents=True, exist_ok=False)
    if mlflow == True:
        setup_mlflow(run_name)
        log_params(locals())
    model, auto_thr = train_model(train_path, rpath, ppath, train_size, batch_size,
                                  balance_train_dataset, balance_val_dataset, balance_snow, train_img,
                                  bn_momentum, learning_rate, stopping_patience, epochs, mlflow)
    print("Finished training and validation, starting evaluation.", flush=True)
    print(f'Working dir: {os.getcwd()}, artifacts dir: {rpath}', flush=True)
    thr = auto_thr if thr is None else thr
    metrics_38Cloud = test_38Cloud(model, thr, C38_path, C38_gtpath, vpath,
                                   rpath / "38Cloud_vis", vids, batch_size)
    mean_metrics_38Cloud = {}
    mean_metrics_38Cloud_snow = {}

    for key, value in metrics_38Cloud.items():
        mean_metrics_38Cloud[key] = np.mean(list(value.values()))
        mean_metrics_38Cloud_snow[f"snow_{key}"] = np.mean([value[x] for x in snow_imgs_38Cloud])

    if mlflow == True:
        log_param('threshold', thr)
        log_metrics(mean_metrics_38Cloud)
        log_metrics(mean_metrics_38Cloud_snow)

    metrics_L8CCA = test_L8CCA(model, thr, L8CCA_path, rpath / "L8CCA_vis", vids, batch_size)
    mean_metrics_L8CCA = {}
    mean_metrics_L8CCA_snow = {}

    for key, value in metrics_L8CCA.items():
        mean_metrics_L8CCA[key] = np.mean(list(value.values()))
        mean_metrics_L8CCA_snow[f"snow_{key}"] = np.mean([value[x] for x in snow_imgs_L8CCA])

    if mlflow == True:
        log_metrics(mean_metrics_L8CCA)
        log_metrics(mean_metrics_L8CCA_snow)
        log_artifacts(rpath)
        end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)
    parser.add_argument("-c", help="config path", default="cloud_detection/cfg/exp_1.yml")
    args = parser.parse_args()

    with open(args.c, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["exp_cfg"]["run_name"] = args.n
    cfg["exp_cfg"]["mlflow"] = args.f

    main(**cfg["exp_cfg"], **cfg["train_cfg"])
