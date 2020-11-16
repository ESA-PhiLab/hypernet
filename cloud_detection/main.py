""" Train and test the models for cloud detection. """

import numpy as np
import argparse
from pathlib import Path
from mlflow import log_metrics, log_artifacts

from train_model import train_model
from evaluate_model import evaluate_model, visualise_model
from utils import setup_mlflow


def main(c):
    if c["mlflow"] == True:
        setup_mlflow(c)
    model = train_model(c["train_path"], c["train_size"], c["batch_size"],
                        c["bn_momentum"], c["learning_rate"], c["stopping_patience"],
                        c["steps_per_epoch"], c["epochs"])
    visualise_model(model, c["test_path"], c["gtpath"], c["vpath"], c["vids"], c["batch_size"])
    metrics = evaluate_model(model, c["test_path"], c["gtpath"], c["batch_size"])
    mean_metrics = {}
    for key, value in metrics.items():
        mean_metrics[key] = np.mean(list(value.values()))


    if c["mlflow"] == True:
        log_metrics(mean_metrics)
        log_artifacts("artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", help="enable mlflow reporting", action="store_true")

    args = parser.parse_args()

    params = {
        "train_path": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "test_path": Path("../datasets/clouds/38-Cloud/38-Cloud_test"),
        "gtpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Entire_scene_gts"),
        "vpath": Path("../datasets/clouds/38-Cloud/38-Cloud_test/Natural_False_Color"),
        "vids": ('*'),
        "train_size": 0.8,
        "batch_size": 8,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "steps_per_epoch": 10,
        "stopping_patience": 20,
        "mlflow": args.f
        }

    main(params)
