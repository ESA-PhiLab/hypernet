from time import time
from keras.callbacks import Callback


class TimeHistory(Callback):
    """
    Custom keras callback logging duration of each epoch
    """
    def on_train_begin(self, logs={}):
        self.on_train_begin_time = time()
        self.times = []
        self.average = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.on_train_begin_time)
        self.average.append(time() - self.epoch_time_start)
