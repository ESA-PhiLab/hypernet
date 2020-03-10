import clize

import ml_intuition.data.io as io


def main(*, graph_path: str, dataset_path: str):
    graph = io.load_pb(graph_path)
    