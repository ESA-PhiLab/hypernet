import math
from typing import Iterable, List
import numpy as np
from utils import normalize_to_byte
from gui.colors import distinct_colors


class ByteColorMap:
    def __init__(self):
        self.color_points = dict()
        self.colors = np.zeros((256, 3)).astype(np.uint8)
        self.dirty = True

    def add_point(self, point: float, color: List[int]) -> None:
        self.color_points[point] = color
        self.dirty = True

    def recalc(self) -> None:
        ordered_color_keys = sorted(self.color_points)
        key_index = 0
        last_key_index = len(ordered_color_keys) - 1
        first_color_key = ordered_color_keys[0]
        last_color_key = ordered_color_keys[last_key_index]
        for i in range(0, 256):
            if i <= ordered_color_keys[0]:
                self.colors[i] = self.color_points[first_color_key]
                continue
            if i >= ordered_color_keys[last_key_index]:
                self.colors[i] = self.color_points[last_color_key]
                continue
            if i > ordered_color_keys[key_index + 1] and key_index < last_key_index:
                key_index = key_index + 1

            color_key = ordered_color_keys[key_index]
            next_color_key = ordered_color_keys[key_index + 1]

            red = np.interp(
                i,
                [color_key, next_color_key],
                [self.color_points[color_key][0], self.color_points[next_color_key][0]]
            )
            green = np.interp(
                i,
                [color_key, next_color_key],
                [self.color_points[color_key][1], self.color_points[next_color_key][1]]
            )
            blue = np.interp(
                i,
                [color_key, next_color_key],
                [self.color_points[color_key][2], self.color_points[next_color_key][2]]
            )

            self.colors[i] = [red, green, blue]

        self.dirty = False

    def get(self, point: float) -> List[int]:
        if self.dirty:
            self.recalc()
        return self.colors[point]


class BandMapper:
    def __init__(self, input_data: np.ndarray):
        self.input_data = input_data

    def map_single(self, band_index: int) -> np.ndarray:
        return (
            self.input_data[:, :, band_index]
            .reshape(self.input_data.shape[0], self.input_data.shape[1], 1)
        )

    def map_colors(self, color_map: ByteColorMap, band_index: int) -> np.ndarray:
        input_data = normalize_to_byte(self.map_single(band_index))

        output_data = np.zeros(
            (input_data.shape[0], input_data.shape[1], 3)
        ).astype(np.uint8)

        for y in range(0, input_data.shape[1]):
            for x in range(0, input_data.shape[0]):
                output_data[x][y] = color_map.get(input_data[x][y])

        return output_data

    def map_mixed(
            self,
            red_band_index: int,
            green_band_index: int,
            blue_band_index: int
    ) -> np.ndarray:
        return self.input_data[:, :, [red_band_index, green_band_index, blue_band_index]]

    def map_visible(self, bands_wavelengths: Iterable[float]) -> np.ndarray:
        best_red_index = 0
        best_red_distance = math.inf

        best_green_index = 0
        best_green_distance = math.inf

        best_blue_index = 0
        best_blue_distance = math.inf

        for band_index, wavelength in enumerate(bands_wavelengths):
            red_distance = abs(632.5 - wavelength)
            green_distance = abs(535 - wavelength)
            blue_distance = abs(475 - wavelength)

            if red_distance < best_red_distance:
                best_red_index = band_index
                best_red_distance = red_distance

            if green_distance < best_green_distance:
                best_green_index = band_index
                best_green_distance = green_distance

            if blue_distance < best_blue_distance:
                best_blue_index = band_index
                best_blue_distance = blue_distance

        return self.map_mixed(best_red_index, best_green_index, best_blue_index)


class GroundTruthMapper:
    def __init__(self, input_data: np.ndarray):
        self.input_data = input_data
        self.available_classes = np.unique(self.input_data)
        self.reset_colors()

    def set_color(self, class_index: int, color: List[int]) -> None:
        self.colors[class_index] = color

    def reset_colors(self) -> None:
        self.colors = dict()
        for color_index, class_index in enumerate(self.available_classes):
            self.colors[class_index] = distinct_colors[color_index % len(distinct_colors)]

    def map_image(self) -> np.ndarray:
        image = np.zeros((self.input_data.shape[0], self.input_data.shape[1], 3)).astype(np.uint8)
        for y in range(0, self.input_data.shape[1]):
            for x in range(0, self.input_data.shape[0]):
                class_index = self.input_data[x][y]
                image[x][y] = self.colors[class_index]
        return image
