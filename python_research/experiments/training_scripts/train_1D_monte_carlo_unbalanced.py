import os.path
import argparse
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.utils import TimeHistory
from python_research.experiments.utils import UnbalancedData
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_1d_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--gt_file', type=str,
                        help="Path to the ground truth in .npy format")
    parser.add_argument("--output_dir", type=str, default="train_grids_output",
                        help="Path to the output directory in which artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data will be stored")
    parser.add_argument("--classes_count", type=int, default=0,
                        help="Number of classes in the dataset")
    parser.add_argument("--runs", type=int, default=10,
                        help="How many times to run the validation")
    parser.add_argument("--training_samples", type=int, default=2000,
                        help="Number of train samples to use")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of training batch')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs without improvement on validation score before '
                             'stopping the learning')
    parser.add_argument('--kernels', type=int, default=200,
                        help='Number of kernels in fir convolutional layer')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='Number of epochs without improvement on validation score before '
                             'stopping the learning')
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of training')
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    # Init data
    data = UnbalancedData(args.dataset_file, args.gt_file, args.training_samples)

    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Normalize data
    max_ = np.max(data.x_train) if np.max(data.x_train) > np.max(data.x_val) else np.max(data.x_val)
    min_ = np.min(data.x_train) if np.min(data.x_train) < np.min(data.x_val) else np.min(data.x_val)
    data.x_train = (data.x_train.astype(np.float64) - min_) / (max_ - min_)
    data.x_val = (data.x_val.astype(np.float64) - min_) / (max_ - min_)
    data.x_test = (data.x_test.astype(np.float64) - min_) / (max_ - min_)

    # Build model
    model = build_1d_model((data.x_train.shape[1], 1), args.kernels, args.kernel_size,
                           len(np.unique(data.y)) - 1)

    # Train model
    history = model.fit(x=data.x_train,
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
    clas_distribution_csv.write(",".join(str(data.counts[label]) for label in data.counts) + "\n")
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
