import numpy as np
import tensorflow as tf
import utils


@utils.check_types((tf.keras.Sequential, tf.keras.Model), tf.data.Dataset)
def train(model: tf.keras.Sequential, dataset: tf.data.Dataset):
    print('Pass')


if __name__ == '__main__':
    # Mocked args:
    train(tf.keras.Sequential(), tf.data.Dataset())
