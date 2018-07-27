import os.path
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from python_research.experiments.multiple_feature_learning.utils.keras_custom_callbacks import TimeHistory
from python_research.experiments.multiple_feature_learning.utils.arguments import parse_single_feature
from python_research.experiments.multiple_feature_learning.utils.dataset import Dataset
from python_research.experiments.multiple_feature_learning.builders.keras_builders import build_single_feature_model, build_settings_for_dataset


def main():
    args = parse_single_feature()
    data = Dataset(args.data_path,
                   args.gt_path,
                   args.nb_samples,
                   args.neighbourhood)

    early = EarlyStopping(patience=args.patience)
    logger = CSVLogger(os.path.join(args.output_dir, args.output_name) + ".csv")
    checkpoint = ModelCheckpoint(os.path.join(args.output_dir, args.output_name) + "_model",
                                 save_best_only=True)
    timer = TimeHistory()

    settings = build_settings_for_dataset(args.dataset_name)

    model = build_single_feature_model(settings,
                                       len(np.unique(data.y)) - 1,
                                       data.x.shape[-1])

    model.fit(x=data.x_train,
              y=data.y_train,
              batch_size=args.batch_size,
              epochs=200,
              verbose=args.verbosity,
              callbacks=[early, logger, checkpoint, timer],
              validation_data=(data.x_val, data.y_val))

    model = load_model(os.path.join(args.output_dir, args.output_name) + "_model")

    print(model.evaluate(x=data.x_test,
                         y=data.y_test))
    times = timer.times
    np.savetxt(os.path.join(args.output_dir, args.output_name) + "_times.csv",
               times,
               fmt="%1.4f")


if __name__ == "__main__":
    main()
