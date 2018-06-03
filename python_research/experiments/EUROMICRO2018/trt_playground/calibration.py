import glob
from random import shuffle

import numpy as np

from experiments.EUROMICRO2018.trt_playground import calibration_helpers

# import tensorrt as trt

MEAN = (71.60167789, 82.09696889, 72.30508881)
MODEL_DIR = '/data/fcn8s/'
CITYSCAPES_DIR = '/data/Cityscapes/'
TEST_IMAGE = CITYSCAPES_DIR + 'leftImg8bit/val/lindau/lindau_000042_000019_leftImg8bit.png'
CALIBRATION_DATASET_LOC = CITYSCAPES_DIR + 'leftImg8bit/train/*/*.png'

CLASSES = 19
CHANNEL = 3
HEIGHT = 512
WIDTH = 1024


def sub_mean_chw(data):
    data = data.transpose((1, 2, 0))  # CHW -> HWC
    data -= np.array(MEAN)  # Broadcast subtract
    data = data.transpose((2, 0, 1))  # HWC -> CHW
    return data


def create_calibration_dataset():
    # Create list of calibration images (filename)
    # This sample code picks 100 images at random from training set
    calibration_files = glob.glob(CALIBRATION_DATASET_LOC)
    shuffle(calibration_files)
    return calibration_files[:100]


def main():
    # calibration_files = create_calibration_dataset()
    calibration_files = []

    # Process 5 images at a time for calibration
    # This batch size can be different from MaxBatchSize (1 in this example)
    batchstream = calibration_helpers.ImageBatchStream(5, calibration_files, sub_mean_chw)
    int8_calibrator = calibration_helpers.PythonEntropyCalibrator(["data"], batchstream)

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


if __name__ == "__main__":
    main()