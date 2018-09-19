import os
import argparse
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from python_research.experiments.multiple_feature_learning.utils.utils import load_patches
from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import TimeHistory
from python_research.experiments.multiple_feature_learning.builders.keras_builders import \
    build_model, build_settings_for_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Script for grids")
    parser.add_argument('--dir', type=str,
                        help="Path to directory containing all patches along with their respective "
                             "ground truths. This directory should also contain test set.")
    parser.add_argument("--output_dir", type=str, default="train_grids_output",
                        help="Path to the output directory in which artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data will be stored")
    parser.add_argument("--sigma", type=int, default=2,
                        help="Sigma value for gaussian filter")
    parser.add_argument("--classes_count", type=int, default=0,
                        help="Number of classes in the dataset")
    parser.add_argument('--neighbourhood', nargs="+", type=int, default=[1, 1],
                        help="Neighbourhood size of the pixel")
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
    # Load patches
    train_data, test = load_patches(args.dir, args.classes_count, args.neighbourhood)
    train_data.add_gaussian_augmented(args.sigma)
    # Normalize data
    train_data.normalize_data(args.classes_count)
    test.x_test = (test.x_test.astype(np.float64) - train_data.min) / (train_data.max - train_data.min)

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Build model
    settings = build_settings_for_dataset(args.neighbourhood)
    model = build_model(settings, args.classes_count, train_data.x_train.shape[-1])

    # Train model
    history = model.fit(x=train_data.x_train,
                        y=train_data.y_train,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=[early, logger, checkpoint, timer],
                        validation_data=(train_data.x_val, train_data.y_val))

    # Load best model
    model = load_model(os.path.join(args.output_dir, args.output_file) + "_model")

    # Calculate test set score
    test_score = model.evaluate(x=test.x_test, y=test.y_test)

    # Calculate accuracy for each class
    predictions = model.predict(x=test.x_test)
    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(test.y_test, axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=[x for x in range(0, args.classes_count)])
    matrix = matrix / matrix.astype(np.float).sum(axis=1)
    class_accuracy = np.diagonal(matrix)

    # Collect metrics
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = len(history.epoch)
    # Save metrics to CSV files
    metrics = open(os.path.join(args.output_dir, "metrics.csv"), 'a')
    class_accuracy_file = open(os.path.join(args.output_dir, "class_accuracy.csv"), 'a')
    metrics.write(
        str(train_score) + "," + str(val_score) + "," + str(test_score[1]) + "," + str(time) + "," + str(epochs) + "," + str(avg_epoch_time) + "\n")
    class_accuracy_file.write(",".join(str(x) for x in class_accuracy) + "\n")
    metrics.close()
    class_accuracy_file.close()
    np.savetxt(os.path.join(args.output_dir, args.output_file) + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    main(args)
