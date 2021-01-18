""" Train and test the models for cloud detection. """

import uuid
import os
import numpy as np
import argparse
from pathlib import Path
from mlflow import log_metrics, log_artifacts, log_param, end_run

from train_model import train_model
from evaluate_38Cloud import evaluate_model as test_38Cloud
from evaluate_L8CCA import evaluate_model as test_L8CCA
from utils import setup_mlflow


def main(c):
    Path(c["rpath"]).mkdir(parents=True, exist_ok=False)
    if c["mlflow"] == True:
        setup_mlflow(c)
    model, auto_thr = train_model(c["train_path"], c["rpath"] / "best_weights", c["train_size"], c["batch_size"],
                                  c["balance_train_dataset"], c["balance_val_dataset"],
                                  c["bn_momentum"], c["learning_rate"], c["stopping_patience"], c["epochs"])
    print("Finished training and validation, starting evaluation.", flush=True)
    print(f'Working dir: {os.getcwd()}, artifacts dir: {c["rpath"]}', flush=True)
    if c["thr"] is None:
        thr = auto_thr
    else:
        thr = c["thr"]
    metrics_38Cloud = test_38Cloud(model, thr, c["38Cloud_path"], c["38Cloud_gtpath"], c["vpath"],
                                   c["rpath"] / "38Cloud_vis", c["vids"], c["batch_size"])
    mean_metrics_38Cloud = {}
    for key, value in metrics_38Cloud.items():
        mean_metrics_38Cloud[key] = np.mean(list(value.values()))
    if c["mlflow"] == True:
        log_param('threshold', thr)
        log_metrics(mean_metrics_38Cloud)
    metrics_L8CCA = test_L8CCA(model, thr, c["L8CCA_path"], c["rpath"] / "L8CCA_vis", c["vids"], c["batch_size"])
    mean_metrics_L8CCA = {}
    for key, value in metrics_L8CCA.items():
        mean_metrics_L8CCA[key] = np.mean(list(value.values()))
    if c["mlflow"] == True:
        log_metrics(mean_metrics_L8CCA)
        log_artifacts(c["rpath"])
        end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)
    parser.add_argument("-s", help="train/validate split", type=float, default=0.8)

    args = parser.parse_args()

    params = {
        "train_path": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "38Cloud_path": Path("../datasets/clouds/38-Cloud/38-Cloud_test"),
        "38Cloud_gtpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts"),
        "L8CCA_path": Path("../datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial"),
        "vpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Natural_False_Color"),
        "rpath": Path(f"artifacts/{uuid.uuid4().hex}"),
        "vids": ('*'),
        "train_size": args.s,
        "batch_size": 32,
        "balance_train_dataset": False,
        "balance_val_dataset": False,
        "thr": 0.5,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "stopping_patience": 20,
        "mlflow": args.f,
        "run_name": args.n
        }

    main(params)
