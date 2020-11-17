from typing import Tuple

import numpy as np
import mlflow
import mlflow.tensorflow


def overlay_mask(image: np.ndarray, mask: np.ndarray, rgb_color: Tuple[float], overlay_intensity: float=0.001) -> np.ndarray:
    """ Overlay a mask on image for visualization purposes. """
    for i, color in enumerate(rgb_color):
        channel = image[:,:,i]
        channel += overlay_intensity * color *  mask[:,:,0]

    return np.clip(image, 0, 1)


def setup_mlflow(c):
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.tensorflow.autolog(every_n_iter=1)
    mlflow.log_params(c)
