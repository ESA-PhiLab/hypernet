import os.path
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import TimeHistory
from python_research.experiments.multiple_feature_learning.utils.arguments import parse_grids
from python_research.experiments.multiple_feature_learning.utils.dataset import Dataset
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_single_feature_model, build_settings_for_dataset

import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


CLASSES_COUNT = 16


def main(args):
    att = "stddev"
    args.dir = "C:\\Users\MMyller\Documents\datasets\Salinas\grids\preprocessed\\fold_4\\" + att
    patch_files = [x for x in sorted_alphanumeric(os.listdir(args.dir)) if 'patch' in x and 'gt' not in x and x.endswith(".npy")]
    gt_files = [x for x in sorted_alphanumeric(os.listdir(args.dir)) if 'patch' in x and 'gt' in x and x.endswith(".npy")]
    test_paths = [x for x in os.listdir(args.dir) if 'test' in x and x.endswith(".npy")]

    train_data = Dataset(os.path.join(args.dir, patch_files[0]),
                         os.path.join(args.dir, gt_files[0]),
                         -1, args.neighbourhood, classes_count=CLASSES_COUNT, normalize=False,
                         validation_set_portion=0.0)
    for file in range(1, len(patch_files)):
        train_data = train_data + Dataset(os.path.join(args.dir, patch_files[file]),
                                          os.path.join(args.dir, gt_files[file]),
                                          -1, args.neighbourhood, classes_count=CLASSES_COUNT,
                                          normalize=False)
    train_data.normalize_train_test_data()

    test = Dataset(os.path.join(args.dir, test_paths[0]),
                   os.path.join(args.dir, test_paths[1]),
                   0, args.neighbourhood, 0, classes_count=CLASSES_COUNT)
    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_name) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_name) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    settings = build_settings_for_dataset(args.dataset_name)

    model = build_single_feature_model(settings,
                                       CLASSES_COUNT,
                                       test.x.shape[-1])

    history = model.fit(x=train_data.x_train,
                        y=train_data.y_train,
                        batch_size=args.batch_size,
                        epochs=200,
                        verbose=args.verbosity,
                        callbacks=[early, logger, checkpoint, timer],
                        validation_data=(train_data.x_val, train_data.y_val))

    model = load_model(os.path.join(args.output_dir, args.output_name) + "_model")

    test_score = model.evaluate(x=test.x_test,
                                y=test.y_test)
    train_score = max(history.history['acc'])
    val_score = max(history.history['val_acc'])
    times = timer.times
    time = times[-1]
    csv = open(os.path.join(args.output_dir, "metrics.csv"), 'a')
    csv.write(str(train_score)+","+str(val_score)+","+str(test_score[1])+","+str(time)+"\n")
    csv.close()
    np.savetxt(os.path.join(args.output_dir, args.output_name) + "_times.csv",
               times,
               fmt="%1.4f")


if __name__ == "__main__":
    args = parse_grids()
    for j in range(0, 5):
        args.dir = args.dir[:-1] + str(j)
        args.output_dir = args.output_dir[:-1] + str(j)
        for i in range(1, 6):
            args.output_name = args.output_name[:-1] + str(i)
            main(args)