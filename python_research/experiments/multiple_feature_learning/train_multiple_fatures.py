import numpy as np
import os.path
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.utils import TimeHistory
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_multiple_features_model, build_settings_for_dataset
from python_research.experiments.utils import \
    TrainTestIndices
from python_research.experiments.utils import Dataset
from python_research.experiments.utils import parse_multiple_features


def main():

    args = parse_multiple_features()

    original_data = Dataset(args.original_path,
                            args.gt_path,
                            args.nb_samples,
                            args.neighborhood)

    train_test_indices = TrainTestIndices(original_data.train_indices,
                                          original_data.val_indices,
                                          original_data.test_indices)

    area_data = Dataset(args.area_path,
                        args.gt_path,
                        args.nb_samples,
                        args.neighborhood,
                        train_test_indices=train_test_indices)
    stddev_data = Dataset(args.stddev_path,
                          args.gt_path,
                          args.nb_samples,
                          args.neighborhood,
                          train_test_indices=train_test_indices)
    diagonal_data = Dataset(args.diagonal_path,
                            args.gt_path,
                            args.nb_samples,
                            args.neighborhood,
                            train_test_indices=train_test_indices)
    moment_data = Dataset(args.moment_path,
                          args.gt_path,
                          args.nb_samples,
                          args.neighborhood,
                          train_test_indices=train_test_indices)

    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_name) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_name) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    settings = build_settings_for_dataset(args.neighborhood)

    model = build_multiple_features_model(settings,
                                          len(np.unique(original_data.y)) - 1,
                                          original_data.x.shape[-1],
                                          area_data.x.shape[-1],
                                          stddev_data.x.shape[-1],
                                          diagonal_data.x.shape[-1],
                                          moment_data.x.shape[-1])

    model.fit(x=[original_data.x_train,
                 area_data.x_train,
                 stddev_data.x_train,
                 diagonal_data.x_train,
                 moment_data.x_train],
              y=original_data.y_train,
              validation_data=([original_data.x_val,
                                area_data.x_val,
                                stddev_data.x_val,
                                diagonal_data.x_val,
                                moment_data.x_val],
                               original_data.y_val),
              epochs=200,
              batch_size=args.batch_size,
              callbacks=[early,
                         logger,
                         checkpoint,
                         timer],
              verbose=args.verbosity)

    model = load_model(os.path.join(args.output_dir, args.output_name) + "_model")
    print(model.evaluate([original_data.x_test,
                          area_data.x_test,
                          stddev_data.x_test,
                          diagonal_data.x_test,
                          moment_data.x_test], original_data.y_test))
    times = timer.times
    np.savetxt(os.path.join(args.output_dir, args.output_name) + "_times.csv", times, fmt="%1.4f")


if __name__ == "__main__":
    main()
