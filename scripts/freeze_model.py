"""
Freeze the graph and store it in .pb (Protocol Buffers) format
"""
import os
import warnings
import json
import clize

warnings.simplefilter(action='ignore', category=FutureWarning)
from ml_intuition.data.utils import freeze_session
import tensorflow as tf
from tensorflow import keras


def main(*, model_path: str, output_dir: str):
    """
    :param model_path: Path to the model to be saved
    :param output_dir: Directory in which the .pb graph and nodes json will be
                       stored
    """
    keras.backend.set_learning_phase(0)
    loaded_model = keras.models.load_model(model_path)

    input_names = [inp.op.name for inp in loaded_model.inputs]
    output_names = [out.op.name for out in loaded_model.outputs]

    nodes = {'input_node': input_names[0],
             'output_node': output_names[0]}

    nodes = json.dumps(nodes, indent=4, sort_keys=True)

    with open(os.path.join(output_dir, "freeze_input_output_node_name.json"),
              "w+") as f:
        f.write(nodes)

    frozen_graph = freeze_session(keras.backend.get_session(),
                                  output_names=output_names)
    tf.train.write_graph(frozen_graph, output_dir, "frozen_graph.pb",
                         as_text=False)
    print("Frozen model saved at {}".format(output_dir))


if __name__ == '__main__':
    clize.run(main)
