import os.path
import numpy as np
import argparse
import string
from copy import copy
from random import shuffle
import re
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.metrics import confusion_matrix
from python_research.experiments.multiple_feature_learning.utils.dataset import Dataset
from python_research.experiments.multiple_feature_learning.utils.patch_data import PatchData
from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import \
    TimeHistory
from python_research.experiments.multiple_feature_learning.builders.keras_builders import \
    build_multiple_features_model, build_settings_for_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--original_data', type=str)
parser.add_argument('--area_data', type=str)
parser.add_argument('--stddev_data', type=str)
parser.add_argument('--diagonal_data', type=str)
parser.add_argument('--moment_data', type=str)
parser.add_argument('--gt', type=str)

parser.add_argument('--original_test', type=str)
parser.add_argument('--area_test', type=str)
parser.add_argument('--stddev_test', type=str)
parser.add_argument('--diagonal_test', type=str)
parser.add_argument('--moment_test', type=str)
parser.add_argument('--gt_test', type=str)

parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('--neighbourhood', type=int, nargs="+")
parser.add_argument('--batch_size', type=int)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--verbosity', type=int, default=2)


CLASSES_COUNT = 16


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def load_patches(path, val_indices=None):
    patch_files = [x for x in sorted_alphanumeric(os.listdir(path)) if 'patch' in x and 'gt' not in x and x.endswith(".npy")]
    gt_files = [x for x in sorted_alphanumeric(os.listdir(path)) if 'patch' in x and 'gt' in x and x.endswith(".npy")]

    train_data = PatchData(os.path.join(path, patch_files[0]),
                           os.path.join(path, gt_files[0]), args.neighbourhood)
    for file in range(1, len(patch_files)):
        train_data += PatchData(os.path.join(path, patch_files[file]),
                                os.path.join(path, gt_files[file]),
                                args.neighbourhood)
    train_data.normalize_data()
    train_data.train_val_split(0.1, val_indices)
    train_data.y_train = to_categorical(train_data.y_train - 1, CLASSES_COUNT)
    train_data.y_val = to_categorical(train_data.y_val - 1, CLASSES_COUNT)
    return train_data


def main(args):
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    original_data = load_patches(args.original_data)
    area_data = load_patches(args.area_data, original_data.val_indices)
    stddev_data = load_patches(args.stddev_data, original_data.val_indices)
    diagonal_data = load_patches(args.diagonal_data, original_data.val_indices)
    moment_data = load_patches(args.moment_data, original_data.val_indices)

    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_name) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_name) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    settings = build_settings_for_dataset(args.neighbourhood)

    model = build_multiple_features_model(settings,
                                          CLASSES_COUNT,
                                          original_data.x_train.shape[-1],
                                          area_data.x_train.shape[-1],
                                          stddev_data.x_train.shape[-1],
                                          diagonal_data.x_train.shape[-1],
                                          moment_data.x_train.shape[-1])

    history = model.fit(x=[original_data.x_train,
                           area_data.x_train,
                           stddev_data.x_train,
                           diagonal_data.x_train,
                           moment_data.x_train],
                        y=original_data.y_train,
                        validation_data=([original_data.x_val,
                                         area_data.x_val,
                                         stddev_data.x_val,
                                         diagonal_data.x_val,
                                         moment_data.x_val], original_data.y_val),
                        epochs=1,
                        batch_size=args.batch_size,
                        callbacks=[early,
                                   logger,
                                   checkpoint,
                                   timer],
                        verbose=args.verbosity)

    model = load_model(os.path.join(args.output_dir, args.output_name) + "_model")
    original_test = Dataset(args.original_test, args.gt_test, 0, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False)
    area_test = Dataset(args.area_test, args.gt_test, 0, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False)
    stddev_test = Dataset(args.stddev_test, args.gt_test, 0, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False)
    diagonal_test = Dataset(args.diagonal_test, args.gt_test, 0, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False)
    moment_test = Dataset(args.moment_test, args.gt_test, 0, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False)

    original_test.x_test = (original_test.x_test.astype(np.float64) - original_data.min) / (
                original_data.max - original_data.min)
    area_test.x_test = (area_test.x_test.astype(np.float64) - area_data.min) / (
                area_data.max - area_data.min)
    stddev_test.x_test = (stddev_test.x_test.astype(np.float64) - stddev_data.min) / (
                stddev_data.max - stddev_data.min)
    diagonal_test.x_test = (diagonal_test.x_test.astype(np.float64) - diagonal_data.min) / (
                diagonal_data.max - diagonal_data.min)
    moment_test.x_test = (moment_test.x_test.astype(np.float64) - moment_data.min) / (
                moment_data.max - moment_data.min)

    test_score = model.evaluate([original_test.x_test,
                                 area_test.x_test,
                                 stddev_test.x_test,
                                 diagonal_test.x_test,
                                 moment_test.x_test], original_test.y_test)

    predictions = model.predict([original_test.x_test,
                                 area_test.x_test,
                                 stddev_test.x_test,
                                 diagonal_test.x_test,
                                 moment_test.x_test])
    predictions = np.argmax(predictions, axis=1)
    y_true = np.argmax(original_test.y_test, axis=1)
    matrix = confusion_matrix(y_true, predictions, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    matrix = matrix / matrix.astype(np.float).sum(axis=1)
    class_accuracy = np.diagonal(matrix)
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    avg_epoch_time = np.average(np.array(timer.average))
    epochs = history.params['epochs']
    csv = open(os.path.join(args.output_dir, "metrics.csv"), 'a')
    csv_a = open(os.path.join(args.output_dir, "class_accuracy.csv"), 'a')
    csv.write(
        str(train_score) + "," + str(val_score) + "," + str(test_score[1]) + "," + str(time) + "," + str(epochs) + "," + str(avg_epoch_time) + "\n")
    csv_a.write(",".join(str(x) for x in class_accuracy) + "\n")
    csv.close()

    np.savetxt(os.path.join(args.output_dir, args.output_name) + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    args = parser.parse_args()
    for i in range(3, 5):
        args.original_data = re.sub('\d', str(i), args.original_data)
        args.area_data = re.sub('\d', str(i), args.area_data)
        args.stddev_data = re.sub('\d', str(i), args.stddev_data)
        args.diagonal_data = re.sub('\d', str(i), args.diagonal_data)
        args.moment_data = re.sub('\d', str(i), args.moment_data)
        args.gt = re.sub('\d', str(i), args.gt)
        args.original_test = re.sub('\d', str(i), args.original_test)
        args.area_test = re.sub('\d', str(i), args.area_test)
        args.stddev_test = re.sub('\d', str(i), args.stddev_test)
        args.diagonal_test = re.sub('\d', str(i), args.diagonal_test)
        args.moment_test = re.sub('\d', str(i), args.moment_test)
        args.gt_test = re.sub('\d', str(i), args.gt_test)
        args.output_dir = args.output_dir[:-1] + str(i)
        for j in range(0, 5):
            args.output_name = args.output_name[:-1] + str(j)
            main(args)