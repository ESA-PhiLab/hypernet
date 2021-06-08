"""
Train and test the models for cloud detection using panchromatic data.

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

import uuid
import os
import yaml
import argparse
from mlflow import log_artifacts, log_param, log_params, set_tags, end_run
from typing import Tuple, List

from cloud_detection.data_gen import DG_L8CCA
from cloud_detection.train_model import train_model
from cloud_detection.evaluate_model import evaluate_model
from cloud_detection.utils import setup_mlflow, make_paths


def main(
    run_name: str,
    train_ids: List[Tuple[str]],
    test_ids: List[Tuple[str]],
    dpath: str,
    rpath: str,
    vids: Tuple[str],
    mlflow: bool,
    train_size: float,
    snow_imgs: List[str],
    batch_size: int,
    thr: float,
    learning_rate: float,
    bn_momentum: float,
    epochs: int,
    stopping_patience: int,
    bands: Tuple[int] = (8,),
    bands_names: Tuple[str] = ("panchromatic",),
):
    """
    Train and test the U-Net model using L8CCA panchromatic images.

    :param run_name: name of the run.
    :param train_ids: IDs of the training images.
    :param test_ids: IDs of the testing images.
    :param dpath: path to dataset.
    :param rpath: path to directory where results and artifacts should be
                  logged (randomly named directory will be created to store the
                  results).
    :param vids: tuple of ids of images which should be used to create
                 visualisations. If contains '*' visualisations will be
                 created for all images in the datasets.
    :param mlflow: whether to use mlflow
    :param train_size: proportion of the training set
                       (the rest goes to validation set).
    :param snow_imgs: list of snow images IDs for testing.
    :param batch_size: size of generated batches, only one batch is loaded
          to memory at a time.
    :param thr: threshold for determining whether pixels contain the clouds
                (if None, threshold will be determined automatically).
    :param learning_rate: learning rate for training.
    :param bn_momentum: momentum of the batch normalization layer.
    :param epochs: number of epochs.
    :param stopping_patience: patience param for early stopping.
    :param bands: band numbers to load
    :param bands_names: names of the bands to load. Should have the same number
                        of elements as bands.
    """
    dpath, rpath = make_paths(dpath, rpath)
    rpath = rpath / uuid.uuid4().hex
    rpath.mkdir(parents=True, exist_ok=False)
    print(f"Working dir: {os.getcwd()}, artifacts dir: {rpath}", flush=True)
    if mlflow:
        setup_mlflow(run_name)
        params = dict(locals())
        del params["train_ids"]
        del params["test_ids"]
        log_params(params)
        set_tags({"train_ids": train_ids,
                  "test_ids": test_ids})

    train_paths = [dpath / id_[0] / id_[1] for id_ in train_ids]
    traingen = DG_L8CCA(img_paths=train_paths, batch_size=batch_size,
                        data_part=(0., train_size), with_gt=True,
                        bands=bands, bands_names=bands_names,
                        resize=True, normalize=False, standardize=True,
                        shuffle=True)
    valgen = DG_L8CCA(img_paths=train_paths, batch_size=batch_size,
                      data_part=(train_size, 1.), with_gt=True,
                      bands=bands, bands_names=bands_names,
                      resize=True, normalize=False, standardize=True,
                      shuffle=True)

    model, auto_thr = train_model(
        traingen=traingen,
        valgen=valgen,
        rpath=rpath,
        bn_momentum=bn_momentum,
        learning_rate=learning_rate,
        stopping_patience=stopping_patience,
        epochs=epochs,
        mlflow=mlflow,
    )
    print("Finished training and validation, starting evaluation.", flush=True)
    thr = auto_thr if thr is None else thr
    evaluate_model(
        dataset_name="L8CCA",
        model=model,
        thr=thr,
        dpath=dpath,
        rpath=rpath / "vis",
        vids=vids,
        batch_size=batch_size,
        img_ids=[id_[1] for id_ in test_ids],
        snow_imgs=snow_imgs,
        mlflow=mlflow,
        bands=bands,
        bands_names=bands_names,
        resize=True,
        normalize=False,
        standardize=True
    )

    if mlflow:
        log_param("threshold", thr)
        log_artifacts(rpath)
        end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", help="enable mlflow reporting", action="store_true")
    parser.add_argument("-n", help="mlflow run name", default=None)
    parser.add_argument(
        "-c", help="config path",
        default="cloud_detection/cfg/exp_panchromatic.yml"
    )
    args = parser.parse_args()

    with open(args.c, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["exp_cfg"]["run_name"] = args.n
    cfg["exp_cfg"]["mlflow"] = args.f

    main(**cfg["exp_cfg"], **cfg["train_cfg"])
