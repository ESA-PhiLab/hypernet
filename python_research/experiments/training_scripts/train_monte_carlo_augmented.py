"""
Script load training a model and evaluating the accuracy on 1D or 3D data
using Monte Carlo validation method. Data can be either balanced (each class
has the same number of samples) or unbalanced (samples drawn randomly)
"""
import os.path
import argparse
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.utils.keras_custom_callbacks import TimeHistory
from python_research.experiments.utils.datasets.subset import BalancedSubset, ImbalancedSubset, CustomSizeSubset
from python_research.experiments.utils.datasets.hyperspectral_dataset import HyperspectralDataset
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_1d_model, build_3d_model, build_settings_for_dataset
from python_research.experiments.utils.io import save_to_csv
from python_research.augmentation.online_augmenter import OnlineAugmenter
from python_research.augmentation.transformations import PCATransformation
from python_research.augmentation.augmenter import Augmenter
from utils import calculate_class_accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help="Path to the dataset in .npy format")
    parser.add_argument('--gt_path', type=str,
                        help="Path to the ground truth in .npy format")
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
    parser.add_argument("--runs", type=int, default=10,
                        help="How many times to run the validation")
    parser.add_argument("--balanced", type=int, default=1,
                        help="Whether each class should have an equal "
                             "number of samples. If True, parameter "
                             "train_samples should be equal to a number of "
                             "samples for each class, if False, paramter "
                             "train_samples should be equal to total number "
                             "of samples in the extracted dataset")
    parser.add_argument("--train_samples", type=int, default=250,
                        help="Number of train samples per class to use")
    parser.add_argument("--pixel_neighbourhood", type=int, default=1,
                        help="Neighbourhood of an extracted pixel when "
                             "preparing the data for training and "
                             "classification. This value should define height "
                             "and width simultaneously.  If equals 1, "
                             "only spectral information will be included "
                             "in a sample")
    parser.add_argument("--sampling_mode", type=str, default="max_twice",
                        help="Type of sampling for augmentation")
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
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.artifacts_path), exist_ok=True)
    # Init data
    test_data = HyperspectralDataset(args.dataset_path, args.gt_path,
                                     neighbourhood_size=args.pixel_neighbourhood)
    test_data.normalize_labels()
    if args.balanced == 1:
        train_data = BalancedSubset(test_data, args.train_samples)
        val_data = BalancedSubset(train_data, args.val_set_part)
    elif args.balanced == 0:
        train_data = ImbalancedSubset(test_data, args.train_samples)
        val_data = ImbalancedSubset(train_data, args.val_set_part)
    elif args.balanced == 2:  # Case for balanced indiana
        train_data = CustomSizeSubset(test_data, [30, 250, 250, 150, 250, 250,
                                                  20, 250, 15, 250, 250, 250,
                                                  150, 250, 50, 50])
        val_data = BalancedSubset(train_data, args.val_set_part)
    # Callbacks
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.artifacts_path, args.output_file) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.artifacts_path, args.output_file) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    # Normalize data
    max_ = train_data.max if train_data.max > val_data.max else val_data.max
    min_ = train_data.min if train_data.min < val_data.min else val_data.min
    train_data.normalize_min_max(min_=min_, max_=max_)
    val_data.normalize_min_max(min_=min_, max_=max_)
    test_data.normalize_min_max(min_=min_, max_=max_)

    # Augment data
    transformation = PCATransformation(low=0.9,
                                       high=1.1,
                                       n_components=train_data.shape[-1])
    transformation.fit(train_data.get_data())
    augmenter = Augmenter(transformation, sampling_mode=args.sampling_mode)
    augmented_data, augmented_labels = augmenter.augment(train_data,
                                                         transformations=1)
    train_data.vstack(augmented_data)
    train_data.hstack(augmented_labels)

    if args.pixel_neighbourhood == 1:
        test_data.expand_dims(axis=-1)
        train_data.expand_dims(axis=-1)
        val_data.expand_dims(axis=-1)

    if args.classes_count == 0:
        args.classes_count = len(np.unique(test_data.get_labels()))

    # Build model
    if args.pixel_neighbourhood == 1:
        model = build_1d_model((test_data.shape[1:]), args.kernels,
                               args.kernel_size, args.classes_count)
    else:
        settings = build_settings_for_dataset((args.pixel_neighbourhood,
                                               args.pixel_neighbourhood))
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

    # Save metrics
    metrics_path = os.path.join(args.artifacts_path, "metrics.csv")
    save_to_csv(metrics_path, [train_score, val_score,
                               test_score[1], time, epochs, avg_epoch_time])
    class_accuracy_path = os.path.join(args.artifacts_path,
                                       "class_accuracy.csv")
    save_to_csv(class_accuracy_path, class_accuracy)
    np.savetxt(os.path.join(args.artifacts_path, args.output_file) +
               "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parse_args()
    for i in range(0, args.runs):
        if i < 10:
            args.output_file = args.output_file[:-1] + str(i)
        else:
            args.output_file = args.output_file[:-2] + str(i)
        main(args)
