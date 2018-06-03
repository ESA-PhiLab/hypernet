import os
from keras.models import load_model
import keras.backend as K
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import freeze_graph
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.training import saver as saver_lib


def convert_keras_to_pb(keras_model_path: str, output_layer_name: str, models_dir: str,
                        model_filename: str) -> None:
    """
    Converts keras model into a protobuf file.
    :param keras_model_path: Path to the stored keras model
    :param output_layer_name: name of the output layer
    :param models_dir: directory where to save the pb file
    :param model_filename: name of the pb file
    :return: None
    """
    model = load_model(keras_model_path)
    K.set_learning_phase(0)
    sess = K.get_session()
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    checkpoint_path = saver.save(sess, os.path.join(models_dir, 'saved_ckpt'),
                                 global_step=0, latest_filename='checkpoint_state')
    graph_io.write_graph(sess.graph, '.', os.path.join(models_dir, 'tmp.pb'))
    freeze_graph.freeze_graph(os.path.join(models_dir, 'tmp.pb'), '',
                              False, checkpoint_path, output_layer_name,
                              "save/restore_all", "save/Const:0",
                              os.path.join(models_dir, model_filename), False, "")
    print("Model saved at:", os.path.join(models_dir, model_filename))


def convert_model(model, config):
    try:
        model.convert_to_pb(config['output_dir'], "model.pb")
        # model.quantize(config['output_dir'], "model.pb")
    except AttributeError:
        model_path = os.path.join(config['output_dir'], 'model.h5')
        model.save(model_path)
        output_layer_name = model.output.name.split(':')[0]
        convert_keras_to_pb(model_path, output_layer_name, config['output_dir'], "model.pb")
