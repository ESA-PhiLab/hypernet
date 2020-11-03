""" Train the models for cloud detection. """

import argparse
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
from tensorflow import keras

from data_gen import DataGenerator, load_image_paths
from models import unet
from losses import jaccard_index, dice_coef, recall, precision, specificity, f1_score


def train_model(dpath, train_size, batch_size, bn_momentum, learning_rate,
                stopping_patience, steps_per_epoch, epochs):
    """
    Train the U-Net model using 38-Cloud dataset.

    :param c: Dict of params.
    """
    # Load data
    train_files, val_files = load_image_paths(
        dpath,
        (train_size, 1-train_size)
        )
    traingen = DataGenerator(
        files=train_files,
        batch_size=batch_size
        )
    valgen = DataGenerator(
        files=val_files,
        batch_size=batch_size
        )

    # Create model
    model = unet(input_size=4,
                bn_momentum=bn_momentum
                )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=jaccard_index(),
        metrics=[
            keras.metrics.binary_crossentropy,
            keras.metrics.binary_accuracy,
            jaccard_index(),
            dice_coef(),
            recall,
            precision,
            specificity,
            f1_score
        ]
    )

    # Prepare training
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=stopping_patience,
            verbose=1
        )
    ]

    # Train model
    model.fit_generator(
        generator=traingen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=valgen,
        validation_steps=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
        )

    # Return model
    return model


if __name__ == "__main__":
    params = {
        "dpath": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "train_size": 0.8,
        "batch_size": 8,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "steps_per_epoch": 10,
        "stopping_patience": 20,
        }
    train_model(**params)
