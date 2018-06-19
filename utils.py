from base64 import b64encode
from io import BytesIO
import imageio
from ipyleaflet import Map, ImageOverlay
import numpy as np
from matplotlib import pyplot as plt


def normalize_to_zero_one(image_data: np.ndarray) -> np.ndarray:
    max_value = image_data.max()
    if max_value == 0:
        return image_data.astype(np.float32)

    return image_data.astype(np.float32) / image_data.max()


def normalize_to_byte(image_data: np.ndarray) -> np.ndarray:
    byte_data = 255 * normalize_to_zero_one(image_data)

    return byte_data.astype(np.uint8)


def serialize_to_url(image_data: np.ndarray) -> str:
    in_memory_file = BytesIO()

    imageio.imwrite(in_memory_file, image_data, format='png')

    ascii_data = b64encode(in_memory_file.getvalue()).decode('ascii')

    return 'data:image/png;base64,' + ascii_data


def create_map(normalized_image: np.ndarray) -> Map:
    width = normalized_image.shape[0]
    height = normalized_image.shape[1]
    bounds = [(-width / 2, -height / 2), (width / 2, height / 2)]

    layer = ImageOverlay(url=serialize_to_url(normalized_image), bounds=bounds)
    leaflet = Map(center=[0, 0], zoom=1, interpolation='nearest')
    leaflet.clear_layers()
    leaflet.add_layer(layer)

    return leaflet


def create_image(normalized_image: np.ndarray, label: str=None):
    plt.figure(figsize=(10, 10))

    if label is not None:
        plt.title(label)

    if normalized_image.shape[2] == 1:
        plt.imshow(np.repeat(normalized_image, 3, axis=2), interpolation='none')
    else:
        plt.imshow(normalized_image, interpolation='none')
