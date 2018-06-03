import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import ctypes
import tensorrt as trt

from experiments.EUROMICRO2018.utils.dataset import Dataset


class PythonEntropyCalibrator(trt.infer.EntropyCalibrator):
    def __init__(self, input_layers, stream):
        trt.infer.EntropyCalibrator.__init__(self)
        self.input_layers = input_layers
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, bindings, names):
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        for i in self.input_layers[0]:
            assert names[0] != i

        bindings[0] = int(self.d_input)
        return bindings

    def read_calibration_cache(self, length):
        return None

    def write_calibration_cache(self, ptr, size):
        cache = ctypes.c_char_p(int(ptr))
        with open('calibration_cache.bin', 'wb') as f:
            f.write(cache.value)
        return None


class ImageBatchStream:
    def __init__(self, batch_size, calibration_files, preprocessor=None):
        self.batch_size = batch_size
        self.max_batches = (len(calibration_files) / batch_size) + \
                           (1 if (len(calibration_files) % batch_size) else 0)
        self.files = calibration_files
        CHANNEL, HEIGHT, WIDTH = calibration_files[0].shape
        self.calibration_data = np.zeros((batch_size, CHANNEL, HEIGHT, WIDTH),
                                         dtype=np.float32)
        self.batch = 0
        self.preprocessor = preprocessor

    @staticmethod
    def read_image_chw(path):
        # img = Image.open(path).resize((WIDTH, HEIGHT), Image.NEAREST)
        # im = np.array(img, dtype=np.float32, order='C')
        # im = im[:, :, ::-1]
        # im = im.transpose((2, 0, 1))
        # return im
        pass

    def reset(self):
        self.batch = 0

    def next_batch(self):
        if self.batch < self.max_batches:
            imgs = []
            files_for_batch = self.files[self.batch_size * self.batch:self.batch_size * (self.batch + 1)]
            for f in files_for_batch:
                print("[ImageBatchStream] Processing ", f)
                # img = ImageBatchStream.read_image_chw(f)
                # img = self.preprocessor(img)
                img = f
                imgs.append(img)
            for i in range(len(imgs)):
                self.calibration_data[i] = imgs[i]
            self.batch += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])


def get_calibration_data(dataset: str):
    dataset = Dataset(dataset, 1)
    dataset.scale_data()
    dataset.reshape_data()
    calibration_set = dataset.x_train
    return calibration_set
