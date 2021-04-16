""" Train the models for cloud detection. """

import time
from pathlib import Path
from mlflow import log_param
from tensorflow import keras

from cloud_detection.models import unet
from cloud_detection.losses import (
    JaccardIndexLoss,
    JaccardIndexMetric,
    DiceCoefMetric,
    recall,
    precision,
    specificity,
)
from cloud_detection.validate import make_validation_insights
from cloud_detection.utils import MLFlowCallback


def train_model(
    traingen: keras.utils.Sequence,
    valgen: keras.utils.Sequence,
    rpath: Path,
    bn_momentum: float,
    learning_rate: float,
    stopping_patience: int,
    epochs: int,
    mlflow: bool,
) -> keras.Model:
    """
    Train the U-Net model using 38-Cloud dataset.

    :param traingen: training set generator.
    :param valgen: validation set generator.
    :param rpath: path to directory where results and
                  artifacts should be logged.
    :param bn_momentum: momentum of the batch normalization layer.
    :param learning_rate: learning rate for training.
    :param stopping_patience: patience param for early stopping.
    :param epochs: number of epochs.
    :param mlflow: whether to use mlflow
    :return: trained model.
    """
    # Create model
    model = unet(input_size=traingen.n_bands, bn_momentum=bn_momentum)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=learning_rate),
        loss=JaccardIndexLoss(),
        metrics=[
            keras.metrics.binary_crossentropy,
            keras.metrics.binary_accuracy,
            JaccardIndexLoss(),
            JaccardIndexMetric(),
            DiceCoefMetric(),
            recall,
            precision,
            specificity,
        ],
    )
    # Save init model to enable independent evaluation later.
    # After update to TF 2, remove this and save and load whole
    # models using keras.callbacks.ModelCheckpoint (now only weights
    # are saved, because TF 1 do not save custom metrics when saving
    # models).
    Path(rpath / "init_model" / "data").mkdir(parents=True, exist_ok=False)
    model.save(rpath / "init_model" / "data" / "model.h5")

    # Prepare training
    Path(rpath / "best_weights").mkdir(parents=True, exist_ok=False)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=stopping_patience, verbose=2),
        keras.callbacks.ModelCheckpoint(
            filepath=f"{rpath}/best_weights/best_weights",
            save_best_only=True,
            save_weights_only=True,
            verbose=2,
        ),
    ]
    if mlflow:
        callbacks.append(MLFlowCallback())

    # Train model
    tbeg = time.time()
    model.fit_generator(
        generator=traingen,
        epochs=epochs,
        validation_data=valgen,
        callbacks=callbacks,
        verbose=2,
    )
    train_time = time.time() - tbeg
    if mlflow:
        log_param("train_time", train_time)
    print(f"Training took { train_time } seconds")
    print("Finished fitting. Will make validation insights now.", flush=True)

    # Load best weights
    model.load_weights(f"{rpath}/best_weights/best_weights")

    # Save validation insights
    tbeg = time.time()
    best_thr = make_validation_insights(
        model, valgen, rpath / "validation_insight")
    val_time = time.time() - tbeg
    if mlflow:
        log_param("val_time", val_time)
    print(f"Generating validation insights took { val_time } seconds")

    # Return model
    return model, best_thr
