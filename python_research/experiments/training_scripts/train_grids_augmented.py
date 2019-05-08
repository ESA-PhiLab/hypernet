"""
Script load training a model and evaluating the accuracy on 1D or 3D data
using already extracted patches (grids).
"""

import os.path
import argparse
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import cohen_kappa_score

from python_research.augmentation.offlin_eaugmenter import OfflineAugmenter
from python_research.augmentation.transformations import *
from python_research.keras_custom_callbacks import TimeHistory
from python_research.dataset_structures import BalancedSubset
from python_research.keras_models import build_1d_model, build_3d_model, build_settings_for_dataset
from utils import calculate_class_accuracy, load_patches
from python_research.io import save_to_csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patches_dir', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument("--artifacts_path", type=str, default="artifacts",
                        help="Path to the output directory in which "
                             "artifacts will be stored")
    parser.add_argument("--output_file", type=str, default="run01",
                        help="Name of the output file in which data "
                             "will be stored")
    parser.add_argument("--classes_count", type=int, default=0,
                        help="Number of classes in the dataset")
    parser.add_argument("--val_set_part", type=float, default=0.1,
                        help="Percentage of a training set to be extracted "
                             "as a validation set")
    parser.add_argument("--pixel_neighborhood", type=int, default=1,
                        help="neighborhood of an extracted pixel when "
                             "preparing the data for training and "
                             "classification. This value should define height "
                             "and width simultaneously.  If equals 1, "
                             "only spectral information will be included "
                             "in a sample")
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Size of training batch')
    parser.add_argument('--patience', type=int, default=15,
                        help='Number of epochs without improvement on '
                             'validation score before '
                             'stopping the learning')
    parser.add_argument('--kernels', type=int, default=200,
                        help='Number of kernels in first convolution layer '
                             '(only for 1D model)')
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Size of a kernel in first convolution layer "
                             "(only for 1D model)")
    parser.add_argument('--verbose', type=int, default=2,
                        help='Verbosity of training')
    parser.add_argument('--start', type=int, default=0,
                        help="number of fold from which to start")
    parser.add_argument('--stop', type=int, default=5,
                        help="number of fold to stop on")
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.artifacts_path), exist_ok=True)
    # Init data
    train_data, test_data = load_patches(args.patches_dir,
                                         args.pixel_neighborhood)
    train_data.normalize_labels()
    test_data.normalize_labels()
    val_data = BalancedSubset(train_data, args.val_set_part)
    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.artifacts_path, args.output_file) +
                       ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.artifacts_path,
                                              args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Normalize data
    max_ = train_data.max if train_data.max > val_data.max else val_data.max
    min_ = train_data.min if train_data.min < val_data.min else val_data.min
    train_data.normalize_min_max(min_=min_, max_=max_)
    val_data.normalize_min_max(min_=min_, max_=max_)
    test_data.normalize_min_max(min_=min_, max_=max_)

    # Augment data
    transformation = UpScaleTransform()
    augmenter = OfflineAugmenter(transformation, sampling_mode="max_twice")
    augmented_data, augmented_labels = augmenter.augment(train_data,
                                                         transformations=1)
    train_data.vstack(augmented_data)
    train_data.hstack(augmented_labels)

    if args.pixel_neighborhood == 1:
        train_data.expand_dims(axis=-1)
        test_data.expand_dims(axis=-1)
        val_data.expand_dims(axis=-1)

    if args.classes_count == 0:
        args.classes_count = len(np.unique(test_data.get_labels()))

    # Build model
    if args.pixel_neighborhood == 1:
        model = build_1d_model((test_data.shape[1:]), args.kernels,
                               args.kernel_size, args.classes_count)
    else:
        settings = build_settings_for_dataset((args.pixel_neighborhood,
                                               args.pixel_neighborhood))
        model = build_3d_model(settings, args.classes_count, test_data.shape[-1])

    # Train model
    history = model.fit(x=train_data.get_data(),
                        y=train_data.get_one_hot_labels(args.classes_count),
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        verbose=args.verbose,
                        callbacks=[early, logger, checkpoint, timer],
                        validation_data=(val_data.get_data(),
                                         val_data.get_one_hot_labels(args.classes_count)))

    # Load best model
    model = load_model(os.path.join(args.artifacts_path, args.output_file) + "_model")

    # Calculate test set score
    test_score = model.evaluate(x=test_data.get_data(),
                                y=test_data.get_one_hot_labels(args.classes_count))

    # Calculate accuracy for each class
    predictions = model.predict(x=test_data.get_data())
    predictions = np.argmax(predictions, axis=1)
    class_accuracy = calculate_class_accuracy(predictions,
                                              test_data.get_labels(),
                                              args.classes_count)
    # Collect metrics
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = len(history.epoch)
    kappa = cohen_kappa_score(predictions, test_data.get_labels())
    # Save metrics
    metrics_path = os.path.join(args.artifacts_path, "metrics.csv")
    kappa_path = os.path.join(args.artifacts_path, "kappa.csv")
    save_to_csv(metrics_path, [train_score, val_score,
                               test_score[1], time, epochs, avg_epoch_time])
    class_accuracy_path = os.path.join(args.artifacts_path, "class_accuracy.csv")
    save_to_csv(class_accuracy_path, class_accuracy)
    save_to_csv(kappa_path, [kappa])
    np.savetxt(os.path.join(args.artifacts_path, args.output_file) +
               "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    for j in range(args.start, args.stop):
        args.patches_dir = args.patches_dir[:-1] + str(j)
        args.artifacts_path = args.artifacts_path[:-1] + str(j)
        for i in range(1, 6):
            args.output_file = args.output_file[:-1] + str(i)
            main(args)
