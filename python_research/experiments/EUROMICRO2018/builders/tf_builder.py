"""Builder for tensorflow models"""
import os
from typing import Tuple, List
import numpy as np
import tensorflow as tf


class History:
    """History stub for tensorflow"""
    def __init__(self):
        self.history = dict()


def get_next_batch(data, labels, batch_size):
    """Batch iterator (unused yet)"""
    for i in range(int(data.shape[0]/batch_size)+1):
        try:
            yield data[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size]
        except IndexError:
            yield data[i * batch_size:], labels[i * batch_size:]


class TensorFlowModel:
    """Wrapper around a tensorflow model with Keras API"""
    def __init__(self, build_model_function, input_shape, n_classes):
        if type(input_shape) != list:
            input_shape = list(input_shape)
        self.x = tf.placeholder(tf.float32, shape=[None]+input_shape)
        self.y = tf.placeholder(tf.float32, shape=[None, n_classes])

        self.output_layer_name = 'my_output0/Softmax'
        self.input_layer_name = 'Placeholder'
        self.prediction = build_model_function(self.x, n_classes, self.output_layer_name)
        self.loss_function = tf.losses.mean_squared_error(self.prediction, self.y)
        self.train_op = None
        self.saver = tf.train.Saver(tf.global_variables())
        self.stored_session = None
        self.output_directory = None
        self.checkpoint_path = None
        self.validation_loss = np.inf
        self.validation_accuracy = 0.0

    def fit(self, data: np.ndarray, labels: np.ndarray, epochs: int,
            batch_size: int, validation_data: Tuple[np.ndarray, np.ndarray],
            verbose: int, callbacks: List) -> History:
        """
        Fit (train) the model
        :param data: data for training
        :param labels: labels for training
        :param epochs: number of epochs
        :param batch_size: size of the batch (unused)
        :param validation_data: (x_val, y_val)
        :param verbose: verbosity
        :param callbacks: Keras callback object
        :return:
        """
        self.output_directory = callbacks[0].filepath.replace("weights.hdf5", '')
        monitored_value = callbacks[0].monitor

        session_history = History()
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss_function)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, epochs+1):
                # for data, labels in get_next_batch(data, labels, batch_size):
                _, train_loss = sess.run([self.train_op, self.loss_function],
                                         feed_dict={self.x: data, self.y: labels})

                train_accuracy = self.calculate_accuracy({self.x: data, self.y: labels})

                validation_accuracy = self.calculate_accuracy({self.x: validation_data[0],
                                                               self.y: validation_data[1]})
                validation_loss = sess.run([self.loss_function], {self.x: validation_data[0],
                                                                  self.y: validation_data[1]})

                if self.model_improved(monitored_value, validation_accuracy, validation_loss):
                    self.checkpoint_path = self.saver.save(sess, os.path.join(self.output_directory, "model.chkpt"))

                # if verbose == 2:
                #     print("Epoch {}/{}:\nLoss: {:.4f}, Train Accuracy: {:.4f}, Validation Accuracy {:.4f}."
                #           .format(i, epochs, loss_value, train_accuracy, validation_accuracy))

                session_history.history[i] = {"Loss": train_loss,
                                              "Accuracy": validation_accuracy}
                if verbose == 1:
                    print("Epoch {}/{}:\nLoss: {:.6f}, "
                          "Train Accuracy: {:.4f}, Validation Accuracy {:.4f}."
                          .format(i, epochs, train_loss, train_accuracy, validation_accuracy))

        return session_history

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
        """
        Starts a tf.Session and calculates loss value and metrics.
        :param data: data
        :param labels: labels
        :return: he loss value and metrics values for the model in test mode.
        """
        with tf.Session() as sess:
            self.saver.restore(sess, os.path.join(self.output_directory, "model.chkpt"))
            loss = sess.run([self.loss_function], feed_dict={self.x: data, self.y: labels})
            accuracy = self.calculate_accuracy({self.x: data, self.y: labels})
            return loss, accuracy

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Runs inference on data.
        :param data: input data
        :return: Predictions.
        """
        with tf.Session() as sess:
            self.saver.restore(sess, os.path.join(self.output_directory, "model.chkpt"))
            predict_op = sess.graph.get_tensor_by_name(self.output_layer_name + ":0")
            prediction = sess.run([predict_op], feed_dict={self.x: data})
            return prediction[0]

    def convert_to_pb(self, models_dir: str, model_filename: str) -> None:
        """
        Converts model to protobuf file
        :param models_dir: directory where to save the model
        :param model_filename: name of the model to be stored
        :return: None
        """
        with tf.Session(graph=tf.Graph()) as sess:
            saver = tf.train.import_meta_graph(self.checkpoint_path + '.meta', clear_devices=True)
            saver.restore(sess, self.checkpoint_path)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, tf.get_default_graph().as_graph_def(),
                [self.output_layer_name])
            with tf.gfile.GFile(os.path.join(models_dir, model_filename), "wb") as file:
                file.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

    # def quantize(self, models_dir: str, model_filename: str, output_name: str="quantized_model.pb") -> None:
    #     """
    #     Calls shell function to quantize the model (TODO)
    #     :param models_dir:
    #     :param model_filename:
    #     :param output_name:
    #     :return:
    #     """
    #     input_model = os.path.join(models_dir, model_filename)
    #     output_model = os.path.join(models_dir, output_name)
    #
    #     commands = ['''~/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph" '''
    #                 '''--in_graph={} --out_graph={} '''
    #                 '''--inputs="input0" --outputs="my_output0/Softmax" '''
    #                 '''--transforms='add_default_attributes strip_unused_nodes(type=float, shape="1,1,103,1") '''
    #                 '''remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) '''
    #                 '''fold_batch_norms fold_old_batch_norms quantize_weights quantize_nodes '''
    #                 '''strip_unused_nodes sort_by_execution_order'"]'''.format(input_model, output_model)]

    def load_from_pb(self, models_dir: str, data: np.ndarray):
        """
        Loads a model from a protobuf file and performs inference on provided data.
        (TODO: remove the inference part and find a way to save the model)
        :param models_dir: directory of the model
        :param data:
        :return:
        """
        with tf.gfile.GFile(models_dir, "rb") as file:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(file.read())

        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name='')

        print("Model loaded from:", models_dir)
        with tf.Session() as sess:
            # self.saver.restore(sess, self.checkpoint_path)
            output_op = sess.graph.get_tensor_by_name(self.output_layer_name+":0")
            # input_op = sess.graph.get_tensor_by_name(self.input_layer_name+":0")

        sess.run(tf.global_variables_initializer())
        prediction = sess.run([output_op], feed_dict={self.x: data})
        # graph_nodes = [n for n in graph_def.node]
        # wts = [n for n in graph_nodes if n.op == 'Const']
        # for n in wts:
        #     a = tensor_util.MakeNdarray(n.attr['value'].tensor)

        return prediction[0]

    def calculate_accuracy(self, feed_dict) -> float:
        """
        Calculates accuracy during tf.Session
        :param feed_dict: dictionary containing data to evaluate
        :return: accuracy
        """
        correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        return accuracy.eval(feed_dict)

    def model_improved(self, monitored_value: str,
                       validation_accuracy: float, validation_loss: float):
        """
        Checks whether the model improved according to monitored value.
        :param monitored_value:
        :param validation_accuracy:
        :param validation_loss:
        :return: (bool) whether the model improved.
        """
        if monitored_value == 'val_acc':
            if self.validation_accuracy < validation_accuracy:
                print("Validation accuracy increased from {:.5f} to {:.5f}. "
                      "Checkpoint saved to: {}".format(self.validation_accuracy,
                                                       validation_accuracy,
                                                       self.output_directory))
                self.validation_accuracy = validation_accuracy
                return True
            return False

        elif monitored_value == 'val_loss':
            if self.validation_loss > validation_loss:
                print("Validation loss decreased from {:.5f} to {:.5f}. "
                      "Checkpoint saved to: {}".format(self.validation_loss,
                                                       validation_loss,
                                                       self.output_directory))
                self.validation_loss = validation_loss
                return True
            return False
        else:
            print("Unknown monitored value.")
            return False
