import os.path
import argparse
import numpy as np
from keras import Model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from typing import NamedTuple, List

from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import TimeHistory
from python_research.experiments.multiple_feature_learning.utils.dataset import Dataset
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_settings_for_dataset, build_model
from python_research.experiments.multiple_feature_learning.builders.keras_builders import (
    build_multiple_features_model,
    build_settings_for_dataset)
from python_research.experiments.multiple_feature_learning.utils.data_types import (
    TrainTestIndices
)



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
    neighbourhood: List[int],
    class_count: int,
    imbalanced=None
) -> TrainingSet:
    settings = build_settings_for_dataset(neighbourhood)

    original_data = Dataset(
        original_path,
        gt_path,
        nb_samples,
        settings.input_neighbourhood,
        imbalanced=imbalanced
    )

    train_test_indices = TrainTestIndices(
        original_data.train_indices,
        original_data.val_indices,
        original_data.test_indices
    )
    original_data.normalize_data()
    bands_sets = [original_data.x.shape[-1]]
    x_trains = [original_data.x_train]
    x_vals = [original_data.x_val]
    x_tests = [original_data.x_test]

    if area_path is not None:
        area_data = Dataset(
            area_path,
            gt_path,
            nb_samples,
            settings.input_neighbourhood,
            train_test_indices=train_test_indices
        )
        area_data.normalize_data()
        bands_sets.append(area_data.x.shape[-1])
        x_trains.append(area_data.x_train)
        x_vals.append(area_data.x_val)
        x_tests.append(area_data.x_test)

    if stddev_path is not None:
        stddev_data = Dataset(
            stddev_path,
            gt_path,
            nb_samples,
            settings.input_neighbourhood,
            train_test_indices=train_test_indices
        )
        stddev_data.normalize_data()
        bands_sets.append(stddev_data.x.shape[-1])
        x_trains.append(stddev_data.x_train)
        x_vals.append(stddev_data.x_val)
        x_tests.append(stddev_data.x_test)

    if diagonal_path is not None:
        diagonal_data = Dataset(
            diagonal_path,
            gt_path,
            nb_samples,
            settings.input_neighbourhood,
            train_test_indices=train_test_indices
        )
        diagonal_data.normalize_data()
        bands_sets.append(diagonal_data.x.shape[-1])
        x_trains.append(diagonal_data.x_train)
        x_vals.append(diagonal_data.x_val)
        x_tests.append(diagonal_data.x_test)

    if moment_path is not None:
        moment_data = Dataset(
            moment_path,
            gt_path,
            nb_samples,
            settings.input_neighbourhood,
            train_test_indices=train_test_indices
        )
        moment_data.normalize_data()
        bands_sets.append(moment_data.x.shape[-1])
        x_trains.append(moment_data.x_train)
        x_vals.append(moment_data.x_val)
        x_tests.append(moment_data.x_test)

    model = build_multiple_features_model(
        settings,
        class_count,
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
    ), original_data.no_train_samples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--area', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--stddev', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--diagonal', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--moment', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--gt_file', type=str,
                        help="Path to the ground truth in .npy format")
    parser.add_argument("--output_dir", type=str, default="train_grids_output",
                        help="Path to the output directory in which artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data will be stored")
    parser.add_argument("--classes_count", type=int, default=0,
                        help="Number of classes in the dataset")
    parser.add_argument("--neighbours_size", nargs="+", type=int, default=[5, 5],
                        help="Neighbourhood of the pixel")
    parser.add_argument("--runs", type=int, default=10,
                        help="How many times to run the validation")
    parser.add_argument("--train_samples_per_class", type=int, default=10,
                        help="Number of train samples to use")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of training batch')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs without improvement on validation score before '
                             'stopping the learning')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of training')
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Init data
    data, train_samples_per_class = build_training_set(args.original, args.gt_file, args.area, args.stddev, args.diagonal,
                              args.moment, args.train_samples_per_class, args.neighbours_size,
                              args.classes_count, imbalanced=1)

    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Train model
    history = data.model.fit(x=data.x_train,
                             y=data.y_train,
                             batch_size=args.batch_size,
                             epochs=args.epochs,
                             verbose=args.verbose,
                             callbacks=[early, logger, checkpoint, timer],
                             validation_data=(data.x_val, data.y_val))

    # Load best model
    model = load_model(os.path.join(args.output_dir, args.output_file) + "_model")

    # Calculate test set score
    test_score = model.evaluate(x=data.x_test,
                                y=data.y_test)

    # Calculate accuracy for each class
    predictions = model.predict(x=data.x_test)
    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(data.y_test, axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=np.unique(y_true))
    matrix = matrix / matrix.astype(np.float).sum(axis=1)
    class_accuracy = np.diagonal(matrix)
    # Collect metrics
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = len(history.epoch)

    # Save metrics
    metrics = open(os.path.join(args.output_dir, "metrics.csv"), 'a')
    class_accuracy_csv = open(os.path.join(args.output_dir, "class_accuracy.csv"), 'a')
    clas_distribution_csv = open(os.path.join(args.output_dir, "class_distribution.csv"), 'a')
    metrics.write(
        str(train_score) + "," + str(val_score) + "," + str(test_score[1]) + "," + str(time) + "," + str(epochs) + "," + str(avg_epoch_time) + "\n")
    class_accuracy_csv.write(",".join(str(x) for x in class_accuracy) + "\n")
    clas_distribution_csv.write(",".join(str(train_samples_per_class[label]) for label in train_samples_per_class) + "\n")
    metrics.close()
    class_accuracy_csv.close()
    clas_distribution_csv.close()

    np.savetxt(os.path.join(args.output_dir, args.output_file) + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    for i in range(0, args.runs):
        if i < 10:
            args.output_file = args.output_file[:-1] + str(i)
        else:
            args.output_file = args.output_file[:-2] + str(i)
        main(args)
