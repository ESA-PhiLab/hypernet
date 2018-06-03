import tensorflow as tf


def uff_hu_tf(input_tensor, n_classes, output_layer_name):
    # reshape = tf.reshape(tensor=input_tensor, shape=[None, 1, 103, 1], name='input0')
    conv1 = tf.layers.conv2d(inputs=input_tensor, filters=100, kernel_size=(1, 11), padding='same', name='input0')
    max_pool = tf.layers.max_pooling2d(conv1, (1, 3), strides=(1, 3), padding='valid')
    flatten = tf.layers.flatten(inputs=max_pool)
    dense = tf.layers.dense(flatten, 100, activation=tf.nn.relu) # pass the first value from iter.get_next() as input
    prediction = tf.layers.dense(dense, n_classes,
                                 activation=tf.nn.softmax,
                                 name=output_layer_name.split("/")[0],
                                 use_bias=False)

    return prediction
