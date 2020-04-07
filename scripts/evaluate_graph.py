import os
import clize
import json
import numpy as np

import tensorflow as tf
import tensorflow.contrib.decent_q
from sklearn import metrics

from ml_intuition.evaluation.performance_metrics import compute_metrics, \
    mean_per_class_accuracy
from ml_intuition.data import io, utils
from ml_intuition import enums
from ml_intuition.evaluation.time_metrics import timeit

METRICS = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    metrics.cohen_kappa_score,
    mean_per_class_accuracy,
    metrics.confusion_matrix
]


def main(*, graph_path: str, node_names_path: str, dataset_path: str):
    graph = io.load_pb(graph_path)
    test_dataset = io.extract_set(dataset_path, enums.Dataset.TEST)

    min_value, max_value = test_dataset[enums.DataStats.MIN], \
                           test_dataset[enums.DataStats.MAX]
    test_data = (test_dataset[enums.Dataset.DATA] - min_value) / \
                (max_value - min_value)

    test_data = np.expand_dims(test_data, axis=-1)

    with open(node_names_path, 'r') as node_names_file:
        node_names = json.loads(node_names_file.read())

    input_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.INPUT] + ':0')
    output_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.OUTPUT] + ':0')

    with tf.Session(graph=graph) as session:
        predict = timeit(session.run)
        predictions, inference_time = predict(output_node,
                                              feed_dict={input_node: test_data})
        predictions = session.run(tf.argmax(predictions, axis=-1))

    graph_metrics = compute_metrics(test_dataset[enums.Dataset.LABELS],
                                    predictions, METRICS)
    graph_metrics['inference_time'] = [inference_time]

    np.savetxt(os.path.join(os.path.dirname(graph_path),
                            metrics.confusion_matrix.__name__ + '.csv'),
               *graph_metrics[metrics.confusion_matrix.__name__],
               delimiter=',',
               fmt='%d')
    del graph_metrics[metrics.confusion_matrix.__name__]

    graph_metrics = utils.restructure_per_class_accuracy(graph_metrics)
    io.save_metrics(dest_path=os.path.dirname(graph_path),
                    file_name='inference_graph_metrics.csv',
                    metrics=graph_metrics)


if __name__ == '__main__':
    clize.run(main)
