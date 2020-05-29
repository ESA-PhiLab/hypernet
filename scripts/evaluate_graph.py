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
from ml_intuition.evaluation.custom_callbacks import timeit
import ml_intuition.data.transforms as transforms

METRICS = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    metrics.cohen_kappa_score,
    mean_per_class_accuracy,
    metrics.confusion_matrix
]


def main(*, graph_path: str, node_names_path: str, dataset_path: str,
         batch_size: int):
    graph = io.load_pb(graph_path)
    test_dict = io.extract_set(dataset_path, enums.Dataset.TEST)
    min_max_path = os.path.join(os.path.dirname(graph_path), "min-max.csv")
    if os.path.exists(min_max_path):
        min_value, max_value = io.read_min_max(min_max_path)

    transformations = [transforms.SpectralTransform(),
                       transforms.MinMaxNormalize(min_=min_value, max_=max_value)]

    test_dict = utils.apply_transformations(test_dict, transformations)

    with open(node_names_path, 'r') as node_names_file:
        node_names = json.loads(node_names_file.read())

    input_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.INPUT] + ':0')
    output_node = graph.get_tensor_by_name(
        node_names[enums.NodeNames.OUTPUT] + ':0')

    with tf.Session(graph=graph) as session:
        predict = timeit(utils.predict_with_graph_in_batches)
        predictions, inference_time = predict(session, input_node, output_node,
                                              test_dict[enums.Dataset.DATA],
                                              batch_size)

    graph_metrics = compute_metrics(test_dict[enums.Dataset.LABELS],
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
