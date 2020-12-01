import os
import clize
import json

import tensorflow as tf
import tensorflow.contrib.decent_q
from sklearn.metrics import confusion_matrix

from ml_intuition.evaluation.performance_metrics import get_model_metrics
from ml_intuition.data import io, utils
from ml_intuition import enums
from ml_intuition.evaluation.time_metrics import timeit
import ml_intuition.data.transforms as transforms


def main(*, graph_path: str, node_names_path: str, dataset_path: str,
         batch_size: int):
    graph = io.load_pb(graph_path)
    test_dict = io.extract_set(dataset_path, enums.Dataset.TEST)
    min_value, max_value = test_dict[enums.DataStats.MIN], \
                           test_dict[enums.DataStats.MAX]

    transformations = [transforms.MinMaxNormalize(min_=min_value,
                                                  max_=max_value),
                       transforms.SpectralTransform()]

    test_dict = transforms.apply_transformations(test_dict, transformations)

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

    graph_metrics = get_model_metrics(test_dict[enums.Dataset.LABELS],
                                      predictions)
    graph_metrics['inference_time'] = [inference_time]
    conf_matrix = confusion_matrix(test_dict[enums.Dataset.LABELS],
                                            predictions)
    io.save_metrics(dest_path=os.path.dirname(graph_path),
                    file_name=enums.Experiment.INFERENCE_GRAPH_METRICS,
                    metrics=graph_metrics)
    io.save_confusion_matrix(conf_matrix, os.path.dirname(graph_path))


if __name__ == '__main__':
    clize.run(main)
