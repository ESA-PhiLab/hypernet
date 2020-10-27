""" Functions to train and test the models for cloud detection. """

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
from typing import Dict
from pathlib import Path
from tensorflow import keras
from data_gen import DataGenerator, load_image_paths
from models import unet

from losses import jaccard_index


def main(c: Dict):
    """
    Train and test the U-Net model using 38-Cloud dataset.

    :param c: Dict of params.
    """
    # Configure MLFlow
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.log_params(c)

    # Load data
    train_files, test_files = load_image_paths(
        c["dpath"],
        (c["train_size"], 1-c["train_size"])
        )
    traingen = DataGenerator(
        files=train_files,
        batch_size=c["batch_size"]
        )
    testgen = DataGenerator(
        files=test_files,
        batch_size=c["batch_size"]
        )

    # Create model
    model = unet(input_size=4,
                bn_momentum=c["bn_momentum"]
                )
    model.compile(
        optimizer=keras.optimizers.Adam(lr=c["learning_rate"]),
        loss=jaccard_index(),
        metrics=[
            keras.metrics.binary_crossentropy,
            keras.metrics.binary_accuracy,
            keras.metrics.AUC(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC()
        ]
    )
    mlflow.tensorflow.autolog(every_n_iter=1)

    # Prepare training
    callbacks = [
        keras.callbacks.EarlyStopping()
    ]

    # Train model
    model.fit_generator(
        generator=traingen,
        steps_per_epoch=c["steps_per_epoch"],
        epochs=c["epochs"],
        validation_data=testgen,
        validation_steps=c["steps_per_epoch"],
        callbacks=callbacks,
        verbose=1
        )

    # # Test model
    # test_loss, test_acc = model.evaluate_generator(
    #     generator=testgen,
    #     steps=c["steps_per_epoch"],
    #     verbose=1
    #     )
    # mlflow.log_metrics({"test loss": test_loss, "test acc": test_acc})
    # # Show sample prediction
    # x, y = next(iter(testgen))
    # y_pred = model.predict(x)
    # for i in range(len(x)):
    #     plt.figure()
    #     plt.imshow(y[i, :, :, 1])
    #     plt.show()
    #     plt.figure()
    #     plt.imshow(y_pred[i, :, :, 1])
    #     plt.show()


if __name__ == "__main__":
    params = {
        "dpath": Path("../datasets/clouds/38-Cloud/38-Cloud_training"),
        "train_size": 0.5,
        "batch_size": 8,
        "learning_rate": .01,
        "bn_momentum": .9,
        "epochs": 200,
        "steps_per_epoch": 10
        }
    main(params)
