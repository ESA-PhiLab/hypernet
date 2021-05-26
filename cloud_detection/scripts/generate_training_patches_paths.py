"""
Script to generate to file training patches paths
for all 5 experiments of cloud detection using RGBNir data.
The generated file can be used to recreate the different
training datasets used for all 5 experiments by loading
appropriate training patches.

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

import numpy as np
from pathlib import Path

from cloud_detection.data_gen import DG_38Cloud
from cloud_detection.utils import load_image_paths


def generate_training_patches_paths(
    train_path: Path = Path("datasets/clouds/38-Cloud/38-Cloud_training"),
    train_size: float = 0.8
):
    """
    Generate to file training patches paths for all 5
    experiments of cloud detection using RGBNir data.

    :param train_path: Path to 38-Cloud training dataset.
    :param train_size: Size of the training set
                       (the rest is used for validation).
    """
    # Exp 1
    T_path = Path("artifacts/T")
    T_path.mkdir(parents=True, exist_ok=False)
    V_path = Path("artifacts/T/V")
    V_path.mkdir(parents=True, exist_ok=False)
    train_files, val_files = load_image_paths(
        base_path=train_path,
        patches_path=None,
        split_ratios=(train_size, 1 - train_size),
        shuffle=True,
        img_id=None,
    )
    print("exp 1", len(train_files), len(val_files))
    for file_ in train_files:
        with open(T_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    for file_ in val_files:
        with open(V_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    # Exp 2a
    T_path = Path("artifacts/T_NS")
    T_path.mkdir(parents=True, exist_ok=False)
    V_path = Path("artifacts/T_NS/V")
    V_path.mkdir(parents=True, exist_ok=False)
    train_img = "LC08_L1TP_002053_20160520_20170324_01_T1"
    train_files, val_files = load_image_paths(
        base_path=train_path,
        patches_path=None,
        split_ratios=(train_size, 1 - train_size),
        shuffle=True,
        img_id=train_img,
    )
    print("exp 2a", len(train_files), len(val_files))
    for file_ in train_files:
        with open(T_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    for file_ in val_files:
        with open(V_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    # Exp 2b
    T_path = Path("artifacts/T_S")
    T_path.mkdir(parents=True, exist_ok=False)
    V_path = Path("artifacts/T_S/V")
    V_path.mkdir(parents=True, exist_ok=False)
    train_img = "LC08_L1TP_035034_20160120_20170224_01_T1"
    train_files, val_files = load_image_paths(
        base_path=train_path,
        patches_path=None,
        split_ratios=(train_size, 1 - train_size),
        shuffle=True,
        img_id=train_img,
    )
    print("exp 2b", len(train_files), len(val_files))
    for file_ in train_files:
        with open(T_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    for file_ in val_files:
        with open(V_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    # Exp 3
    T_path = Path("artifacts/T_C")
    T_path.mkdir(parents=True, exist_ok=False)
    V_path = Path("artifacts/T_C/V")
    V_path.mkdir(parents=True, exist_ok=False)
    train_files, val_files = load_image_paths(
        base_path=train_path,
        patches_path=None,
        split_ratios=(train_size, 1 - train_size),
        shuffle=True,
        img_id=None
    )
    traingen = DG_38Cloud(
        files=train_files,
        batch_size=4,
        balance_classes=False,
        balance_snow=True,
    )
    train_files = list(
        np.array(traingen._files)[np.sort(traingen._file_indexes)])
    print("exp 3", len(train_files), len(val_files))
    for file_ in train_files:
        with open(T_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    for file_ in val_files:
        with open(V_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    # Exp 4
    T_path = Path("artifacts/T'")
    T_path.mkdir(parents=True, exist_ok=False)
    V_path = Path("artifacts/T'/V")
    V_path.mkdir(parents=True, exist_ok=False)
    ppath = "datasets/clouds/38-Cloud/training_patches_38-cloud_nonempty.csv"
    train_files, val_files = load_image_paths(
        base_path=train_path,
        patches_path=ppath,
        split_ratios=(train_size, 1 - train_size),
        shuffle=True,
        img_id=None,
    )
    print("exp 4", len(train_files), len(val_files))
    for file_ in train_files:
        with open(T_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")
    for file_ in val_files:
        with open(V_path / "patches.csv", "a") as f:
            f.write(file_["gt"].parts[-1][3:-4]+"\n")


if __name__ == "__main__":
    generate_training_patches_paths()
