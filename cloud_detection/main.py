""" Train and test the models for cloud detection. """

import uuid
import numpy as np
import argparse
from pathlib import Path
from mlflow import log_metrics, log_artifacts, end_run

from train_model import train_model
from evaluate_38Cloud import evaluate_model as test_38Cloud
from evaluate_L8CCA import evaluate_model as test_L8CCA
from utils import setup_mlflow


def main(c):
    Path(c["rpath"]).mkdir(parents=True, exist_ok=False)
    if c["mlflow"] == True:
        setup_mlflow(c)
    model = train_model(c["train_path"], c["rpath"], c["train_size"], c["batch_size"],
                        c["bn_momentum"], c["learning_rate"], c["stopping_patience"],
                        c["steps_per_epoch"], c["epochs"])
    metrics_38Cloud = test_38Cloud(model, c["38Cloud_path"], c["38Cloud_gtpath"], c["vpath"],
                                   c["rpath"], c["vids"], c["batch_size"])
    mean_metrics_38Cloud = {}
    for key, value in metrics_38Cloud.items():
        mean_metrics_38Cloud[key] = np.mean(list(value.values()))
    if c["mlflow"] == True:
        log_metrics(mean_metrics_38Cloud)
    metrics_L8CCA = test_L8CCA(model, c["L8CCA_path"], c["rpath"], c["vids"], c["batch_size"])
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

    args = parser.parse_args()

    params = {
        "train_path": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "38Cloud_path": Path("../datasets/clouds/38-Cloud/38-Cloud_test"),
        "38Cloud_gtpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts"),
        "L8CCA_path": Path("../datasets/clouds/Landsat-Cloud-Cover-Assessment-Validation-Data-Partial"),
        "vpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Natural_False_Color"),
        "rpath": Path(f"artifacts/{uuid.uuid4().hex}"),
        "vids": ('*'),
        "train_size": 0.8,
        "batch_size": 8,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "steps_per_epoch": 10,
        "stopping_patience": 20,
        "mlflow": args.f,
        "run_name": args.n
        }

    main(params)
