import numpy as np
import os.path
from keras.models import load_model, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.utils import (
    TimeHistory
)
from python_research.experiments.multiple_feature_learning.builders.keras_builders import (
    build_multiple_features_model,
    build_settings_for_dataset
)
from python_research.experiments.utils import (
    TrainTestIndices
)
from python_research.experiments.utils import Dataset
from python_research.experiments.utils import (
    parse_multiple_features
)

from typing import List, NamedTuple
from python_research.validation import validate


class TrainingSet(NamedTuple):
    x_train: list
    x_test: list
    x_val: list
    y_train: list
    y_test: list
    y_val: list
    model: Model


def build_training_set(
    original_path: str,
    gt_path: str,
    area_path: str,
    stddev_path: str,
    diagonal_path: str,
    moment_path: str,
    nb_samples: int,
    neighborhood: List[int]
) -> TrainingSet:
    settings = build_settings_for_dataset(neighborhood)

    original_data = Dataset(
        original_path,
        gt_path,
        nb_samples,
        settings.input_neighborhood
    )

    train_test_indices = TrainTestIndices(
        original_data.train_indices,
        original_data.test_indices
    )

    bands_sets = [original_data.x.shape[-1]]
    x_trains = [original_data.x_train]
    x_vals = [original_data.x_val]
    x_tests = [original_data.x_test]

    if area_path is not None:
        area_data = Dataset(
            area_path,
            gt_path,
            nb_samples,
            settings.input_neighborhood,
            train_test_indices=train_test_indices
        )

        bands_sets.append(area_data.x.shape[-1])
        x_trains.append(area_data.x_train)
        x_vals.append(area_data.x_val)
        x_tests.append(area_data.x_test)

    if stddev_path is not None:
        stddev_data = Dataset(
            stddev_path,
            gt_path,
            nb_samples,
            settings.input_neighborhood,
            train_test_indices=train_test_indices
        )

        bands_sets.append(stddev_data.x.shape[-1])
        x_trains.append(stddev_data.x_train)
        x_vals.append(stddev_data.x_val)
        x_tests.append(stddev_data.x_test)

    if diagonal_path is not None:
        diagonal_data = Dataset(
            diagonal_path,
            gt_path,
            nb_samples,
            settings.input_neighborhood,
            train_test_indices=train_test_indices
        )

        bands_sets.append(diagonal_data.x.shape[-1])
        x_trains.append(diagonal_data.x_train)
        x_vals.append(diagonal_data.x_val)
        x_tests.append(diagonal_data.x_test)

    if moment_path is not None:
        moment_data = Dataset(
            moment_path,
            gt_path,
            nb_samples,
            settings.input_neighborhood,
            train_test_indices=train_test_indices
        )

        bands_sets.append(moment_data.x.shape[-1])
        x_trains.append(moment_data.x_train)
        x_vals.append(moment_data.x_val)
        x_tests.append(moment_data.x_test)

    model = build_multiple_features_model(
        settings,
        len(original_data.labels) - 1,
        bands_sets
    )

    return TrainingSet(
        x_train=x_trains,
        x_test=x_tests,
        x_val=x_vals,
        y_train=original_data.y_train,
        y_test=original_data.y_test,
        y_val=original_data.y_val,
        model=model
    )


def main():
    args = parse_multiple_features()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    training_set = build_training_set(
        args.original_path,
        args.gt_path,
        args.area_path,
        args.stddev_path,
        args.diagonal_path,
        args.moment_path,
        args.nb_samples,
        args.neighborhood
    )

    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(output_path + ".csv")
    checkpoint = ModelCheckpoint(
        output_path + "_model",
        save_best_only=True
    )
    timer = TimeHistory()

    training_set.model.fit(
        x=training_set.x_train,
        y=training_set.y_train,
        validation_data=(training_set.x_val, training_set.y_val),
        epochs=200,
        batch_size=args.batch_size,
        callbacks=[
            early,
            logger,
            checkpoint,
            timer
        ],
        verbose=args.verbosity
    )

    model = load_model(output_path + "_model")
    print(validate(model, training_set))
    times = timer.times
    np.savetxt(output_path + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    main()
