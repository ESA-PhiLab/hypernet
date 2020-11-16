import numpy as np
import mlflow
import mlflow.tensorflow


def overlay_mask(image: np.ndarray, mask: np.ndarray, channel_no, overlay_intensity: float=0.3) -> np.ndarray:
    """ Overlay a mask on image for visualization purposes. """
    mask_channel = image[:,:,channel_no]
    mask_channel += overlay_intensity * mask[:,:,0]
    return np.clip(image, 0, 1)


def setup_mlflow(c):
    mlflow.set_tracking_uri("http://beetle.mlflow.kplabs.pl")
    mlflow.set_experiment("cloud_detection")
    mlflow.tensorflow.autolog(every_n_iter=1)
    mlflow.log_params(c)
