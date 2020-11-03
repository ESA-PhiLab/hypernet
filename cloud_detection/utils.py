import numpy as np
import mlflow
import mlflow.tensorflow


def overlay_mask(image: np.ndarray, mask: np.ndarray, overlay_intensity: float=0.3) -> np.ndarray:
    """ Overlay a mask on image for visualization purposes. """
    ret = np.copy(image)
    red_channel = ret[:,:,0]
    red_channel += overlay_intensity * mask[:,:,0]
    return np.clip(ret, 0, 1)


def setup_mlflow(c):
     mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
     mlflow.set_experiment("cloud_detection")
     mlflow.tensorflow.autolog(every_n_iter=1)
     mlflow.log_params(c)
