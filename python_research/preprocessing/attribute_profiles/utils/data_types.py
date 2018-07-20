import numpy as np
from typing import NamedTuple


class Pixel:

    def __init__(self, x, y, gray_level):
        self.x = np.uint16(x)
        self.y = np.uint16(y)
        self.gray_level = gray_level

    @property
    def coords(self):
        return self.y, self.x

    def __eq__(self, other):
        if type(other) != Pixel:
            return False
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        if type(other) != Pixel:
            return self.gray_level < other
        else:
            return self.gray_level < other.gray_level


class Moments(NamedTuple):
    moment00: float
    moment10: float
    moment01: float
    moment20: float
    moment02: float
    x_mass_center: float
    y_mass_center: float
