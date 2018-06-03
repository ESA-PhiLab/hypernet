from sklearn.preprocessing import LabelBinarizer

from experiments.EUROMICRO2018.trt_playground.calibration_helpers import get_calibration_data
from experiments.EUROMICRO2018.trt_playground.uff_engine import create_int_engine
from experiments.EUROMICRO2018.utils.arguments import parse, print_config
from experiments.EUROMICRO2018.utils.dataset import Dataset
from experiments.EUROMICRO2018.utils.load_data import *


def main():
    np.random.seed(0)
    config = {'epochs': parse().nb_epoch,
              'dataset': parse().dataset_name,
              'batch_size': parse().batch_size,
              'nb_samples': parse().nb_samples,
              'verbosity': parse().verbosity,
              'output_dir': parse().output_dir,
              'local': parse().local,
              'convert': parse().convert}

    print_config(config)
    dataset = Dataset(config['dataset'], config['nb_samples'])
    n_classes = len(dataset.labels)
    dataset.scale_data()
    dataset.reshape_data()
    X_train, y_train, X_test, y_test = dataset.get_data()


    n_channels = X_train.shape[2]
    input_shape = (1, n_channels, 1)

    enc = LabelBinarizer()
    y_train = enc.fit_transform(y_train)
    y_test = enc.transform(y_test)

    calibration_data = get_calibration_data(config['dataset'])

    print("Creating an engine...")
    # engine = create_float_engine(X_test.shape[1:], config['output_dir'])

    # engine = create_int_engine2(calibration_data, config['output_dir'])

    engine = create_int_engine(calibration_data, config['output_dir'])

    # G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
    # print("Engine created.")
    #
    # print("Creating runtime and context...")
    # runtime = trt.infer.create_infer_runtime(G_LOGGER)
    # context = engine.create_execution_context()
    # output = np.empty(n_classes, dtype=np.float32)
    # print("Runtime and context created.")
    #
    # # alocate device memory
    # print("Allocating memory...")
    # d_input = cuda.mem_alloc(1 * X_test[0].size * X_test[0].dtype.itemsize)
    # d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)
    # print("Memory allocated.")
    #
    # bindings = [int(d_input), int(d_output)]
    #
    # stream = cuda.Stream()
    # # transfer input data to device
    # cuda.memcpy_htod_async(d_input, X_test[500], stream)
    # # execute model
    # context.enqueue(1, bindings, stream.handle, None)
    # # transfer predictions back
    # cuda.memcpy_dtoh_async(output, d_output, stream)
    # # syncronize threads
    # stream.synchronize()

    output = engine.infer(X_test[0])
    print("Test Case: " + str("a"))
    print("Prediction proba: ", output)
    print("Prediction: " + str(np.argmax(output)))

    context.destroy()
    engine.destroy()
    runtime.destroy()


if __name__ == "__main__":
    main()