import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, overlay_intensity: float=0.3) -> np.ndarray:
    """ Overlay a mask on image for visualization purposes. """
    ret = np.copy(image)
    red_channel = ret[:,:,0]
    red_channel += overlay_intensity * mask[:,:,0]
    return np.clip(ret, 0, 1)
