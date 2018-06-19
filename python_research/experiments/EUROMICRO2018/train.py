import pickle
from datetime import datetime
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from experiments.EUROMICRO2018.model_conversion.model_converter import convert_model
from experiments.EUROMICRO2018.utils.arguments import parse, print_config
from experiments.EUROMICRO2018.utils.dataset import Dataset
from experiments.EUROMICRO2018.utils.load_data import *

from experiments.EUROMICRO2018.builders.tf_builder import TensorFlowModel
from experiments.EUROMICRO2018.builders.tf_models import uff_hu_tf
from experiments.EUROMICRO2018.builders.keras_builder import uff_hu

'''Local run:
-d pavia -s 0 -e 1 -b 100 -v 1 -l -c -o <output_path>
'''

'''Remote run:
-m titan -i "..\experiments/EUROMICRO2018/train.py" -p "-d indiana -s 0 -e 200 -b 100 -v 1 -c "
-d dataset {indiana, pavia, salinas}
-s amount of samples for training, 0 - all possible
-e number of epochs
-b batch size (only for keras so far)
-v verbosity {0, 1, 2}
-c {--no-c} convert model {do not convert model} to protobuf file .pb
'''


def main():
    np.random.seed(0)
    run_arguments = parse()
    config = {'epochs': run_arguments.nb_epoch,
              'dataset': run_arguments.dataset_name,
              'batch_size': run_arguments.batch_size,
              'nb_samples': run_arguments.nb_samples,
              'verbosity': run_arguments.verbosity,
              'output_dir': run_arguments.output_dir,
              'local': run_arguments.local,
              'convert': run_arguments.convert}

    if config['local']:
        now = '{date:%Y%m%d_%H%M%S}'.format(date=datetime.now())
        config['output_dir'] = os.path.join(config['output_dir'], now)
        if not os.path.exists(config['output_dir']):
            os.makedirs(config['output_dir'])

    print_config(config)
    dataset = Dataset(config['dataset'], config['nb_samples'])
    n_classes = len(dataset.labels)+1
    dataset.scale_data()
    dataset.reshape_data()
    X_train, y_train, X_test, y_test = dataset.get_data()
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.2, random_state=0)

    n_channels = X_train.shape[2]
    input_shape = (1, n_channels, 1)

    # model = uff_hu(input_shape, n_classes)
    model = TensorFlowModel(uff_hu_tf, input_shape, n_classes)

    checkpointer = ModelCheckpoint(filepath=os.path.join(config['output_dir'], 'weights.hdf5'),
                                   verbose=1,
                                   save_best_only=True,
                                   monitor='val_acc',
                                   mode='max',
                                   save_weights_only=True)

    history = model.fit(X_train, y_train,
                        epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        validation_data=(X_validation, y_validation),
                        verbose=config['verbosity'],
                        callbacks=[checkpointer])

    with open(os.path.join(config['output_dir'], 'history.pkl'), 'wb') as file:
        pickle.dump(history.history, file)

    loss, accuracy = model.evaluate(X_test, y_test)
    prediction = model.predict(X_test)
    print('Test Accuracy:', accuracy)

    if config['convert']:
        # model = uff_hu(input_shape, num_classes=n_classes)
        convert_model(model, config)
        # TODO implement automatic quantization


    # path = 'C:\\Ecores\\Hyperspectral\\quantized_model.pb'
    # prediction = model.load_from_pb(os.path.join(config['output_dir'], "quantized_model.pb"), X_test)
    # # prediction = model.load_from_pb(path, X_test)
    # print('Test Accuracy quantized:', accuracy_score(np.argmax(y_test, axis=1), np.argmax(prediction, axis=1)))
    # print()


if __name__ == "__main__":
    main()
