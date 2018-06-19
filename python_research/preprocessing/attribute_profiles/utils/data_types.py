import numpy as np
from typing import NamedTuple


class Pixel:
    def __init__(self, x, y, value):
        self.x = np.uint16(x)
        self.y = np.uint16(y)
        self.value = value

    @property
    def coords(self):
        return self.y, self.x

    def __float__(self):
        return float(self.value)


class Moments(NamedTuple):
    moment00: float
    moment10: float
    moment01: float
    moment20: float
    moment02: float
    x_mass_center: float
    y_mass_center: float
