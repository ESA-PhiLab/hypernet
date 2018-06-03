import os
from typing import Tuple

import numpy as np
import tensorrt as trt
import uff
from keras.models import load_model
from tensorrt.parsers import uffparser

from experiments.EUROMICRO2018.trt_playground import calibration_helpers


def create_float_engine(input_shape: Tuple, model_path: str):
    """
    Creates engine from given frozen model and saves it to disk
    :param input_shape: shape of input data
    :param model_path: path to saved model
    :return: engine
    """
    assert os.path.exists(model_path)

    model = load_model(os.path.join(model_path, 'model.h5'))
    output_layer_name = model.output.name.split(':')[0]
    print(output_layer_name)
    if output_layer_name != 'output_1/Softmax':
        output_layer_name = 'output_1/Softmax'
    input_layer_name = model.input.name.split(':')[0]
    if input_layer_name != 'input0_1':
        input_layer_name = 'input0_1'

    print(input_layer_name)
    image_shape = input_shape
    image_shape = (image_shape[0], image_shape[2], image_shape[1])
    print("Image shape:", image_shape)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    PATH = os.path.join(model_path, 'model.engine')

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(PATH.replace(".engine", ".pb"), [output_layer_name])

    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(input_layer_name, image_shape, 0)
    parser.register_output(output_layer_name)

    # Build your TensorRT inference engine
    # This step performs (1) Tensor fusion (2) Reduced precision
    # (3) Target autotuning (4) Tensor memory management
    engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                         uff_model,
                                         parser,
                                         1, #batch size
                                         1<<20,
                                         trt.infer.DataType.FLOAT)

    parser.destroy()

    return engine


def create_int_engine(calibration_data: np.ndarray, model_path: str):
    """
    Creates engine from given frozen model and saves it to disk
    :param calibration_data: data used for calibration
    :param model_path: path to saved model
    :return: engine
    """

    model = load_model(os.path.join(model_path, 'model.h5'))
    input_layers = [model.input.name.split(':')[0]]
    output_layers = [model.output.name.split(':')[0]]

    if input_layers[0] != 'input0_1':
        input_layers[0] = 'input0_1'
    if output_layers[0] != 'output_0_1/Softmax':
        output_layers[0] = 'output_0_1/Softmax'

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(os.path.join(model_path, 'model.pb'), output_layers)

    calibration_files = calibration_data
    data_size = calibration_data.shape[0]
    sample_shape = calibration_data[0].shape

    # Process 5 images at a time for calibration
    # This batch size can be different from MaxBatchSize (1 in this example)

    batchstream = calibration_helpers.ImageBatchStream(data_size, calibration_files)
    int8_calibrator = calibration_helpers.PythonEntropyCalibrator(input_layers, batchstream)

    # Easy to use TensorRT lite package
    # engine = trt.lite.Engine(framework="c1",
    #                          deployfile=MODEL_DIR + "fcn8s.prototxt",
    #                          modelfile=MODEL_DIR + "fcn8s.caffemodel",
    #                          max_batch_size=1,
    #                          max_workspace_size=(256 << 20),
    #                          input_nodes={"data": (CHANNEL, HEIGHT, WIDTH)},
    #                          output_nodes=["score"],
    #                          preprocessors={"data": sub_mean_chw},
    #                          postprocessors={"score": color_map},
    #                          data_type=trt.infer.DataType.INT8,
    #                          calibrator=int8_calibrator,
    #                          logger_severity=trt.infer.LogSeverity.INFO)
    engine = trt.lite.Engine(framework='tf',
                             path=os.path.join(model_path, 'model.pb'),
                             # stream=uff_model,
                             max_batch_size=100,
                             max_workspace_size=(256 << 20),
                             input_nodes={input_layers[0]: sample_shape},
                             output_nodes=output_layers,
                             data_type=trt.infer.DataType.FLOAT,
                             calibrator=int8_calibrator,
                             logger_severity=trt.infer.LogSeverity.INFO,
                             preprocessors=None,
                             postprocessors=None)

    return engine



def create_int_engine2(calibration_data: np.ndarray, model_path: str):
    """
    Creates engine from given frozen model and saves it to disk
    :param input_shape: shape of input data
    :param model_path: path to saved model
    :return: engine
    """
    assert os.path.exists(model_path)

    model = load_model(os.path.join(model_path, 'model.h5'))
    input_layers = [model.input.name.split(':')[0]]
    output_layers = [model.output.name.split(':')[0]]

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    PATH = os.path.join(model_path, 'model.engine')

    # Load your newly created Tensorflow frozen model and convert it to UFF
    uff_model = uff.from_tensorflow_frozen_model(PATH.replace(".engine", ".pb"), output_layers)

    calibration_files = calibration_data
    data_size = calibration_data.shape[0]
    sample_shape = calibration_data[0].shape

    # Process 5 images at a time for calibration
    # This batch size can be different from MaxBatchSize (1 in this example)

    batchstream = calibration_helpers.ImageBatchStream(data_size, calibration_files)
    int8_calibrator = calibration_helpers.PythonEntropyCalibrator(input_layers, batchstream)


    # Create a UFF parser to parse the UFF file created from your TF Frozen model
    parser = uffparser.create_uff_parser()
    parser.register_input(input_layers[0], sample_shape, 0)
    parser.register_output(output_layers[0])

    # Build your TensorRT inference engine
    # This step performs (1) Tensor fusion (2) Reduced precision
    # (3) Target autotuning (4) Tensor memory management
    engine = trt.utils.uff_to_trt_engine(G_LOGGER,
                                         uff_model,
                                         parser,
                                         8, #batch size
                                         1<<20,
                                         trt.infer.DataType.INT8,
                                         calibrator=int8_calibrator)

    parser.destroy()

    return engine


def serialize_engine(engine, output_path: str):
    """
    Serializes engine to store it on disk
    :param engine: engine to serialize
    :param output_path: path to save the engine
    :return: None
    """
    assert os.path.exists(output_path)

    trt.utils.write_engine_to_file(output_path, engine.serialize())
    print("Model serialized at:", output_path)


def load_engine():
    pass
