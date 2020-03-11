import clize
import h5py
import json
import numpy as np

import tensorflow as tf
import tensorflow.contrib.decent_q
from sklearn.metrics import accuracy_score

import ml_intuition.data.io as io
import ml_intuition.enums as enums


def main(*, graph_path: str, node_names_path: str, dataset_path: str):
    graph = io.load_pb(graph_path)

    with h5py.File(dataset_path, 'r') as file:
        test_data = file[enums.Dataset.TEST][enums.Dataset.DATA][:]
        test_labels = file[enums.Dataset.TEST][enums.Dataset.LABELS][:]
        min_value, max_value = file.attrs[enums.DataStats.MIN], \
                               file.attrs[enums.DataStats.MAX]

    test_data = (test_data - min_value) / (max_value - min_value)
    test_data = np.expand_dims(test_data, axis=-1)

    with open(node_names_path, 'r') as node_names_file:
        node_names = json.loads(node_names_file.read())

    input_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.INPUT] + ':0')
    output_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.OUTPUT] + ':0')

    with tf.Session(graph=graph) as session:
        predictions = session.run(output_node,
                                  feed_dict={input_node: test_data})
        predictions = session.run(tf.argmax(predictions, axis=-1))

    print('acc: {}'.format(accuracy_score(test_labels, predictions)))


if __name__ == '__main__':
    clize.run(main)
