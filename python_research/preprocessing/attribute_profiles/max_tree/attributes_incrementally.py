from math import sqrt
from ..utils.image_moments import spatial_moments, mass_center, central_moments, \
    normalize_central_moments


class IncrementalAttribute:
    def __add__(self, other):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError


class Area(IncrementalAttribute):
    def __init__(self, area=1):
        self.area = area

    def __add__(self, other):
        area = self.area + other.area
        return Area(area)

    def get(self):
        return self.area


class StandardDeviation(IncrementalAttribute):
    mean = 0.0
    n_samples = 1.0
    variance = 0.0

    def __init__(self, value=-1, mean=0, n_samples=0, variance=0):
        if value != -1:
            self.mean = float(value)
            self.variance = pow(float(value), 2)
        else:
            self.mean = mean
            self.n_samples = n_samples
            self.variance = variance

    def __add__(self, other):
        if type(other) == StandardDeviation:
            variance = self._combine_variance(other)
            mean = self._combine_mean(other)
            n_samples = self.n_samples + other.n_samples
            return StandardDeviation(mean=mean, n_samples=n_samples,
                                     variance=variance)
        else:
            raise TypeError(
                "Cannot add object of type StandardDeviation to {}".format(
                    str(type(other))))

    def _combine_mean(self, other):
        return ((self.n_samples * self.mean) + (other.n_samples *
                                                other.mean)) / \
               (self.n_samples + other.n_samples)

    def _combine_variance(self, other):
        combined_mean = self._combine_mean(other)
        combined_variance = 1. / (self.n_samples + other.n_samples - 1.0) * ((
                                                                                     self.n_samples - 1) * self.variance + self.n_samples *
                                                                             pow(
                                                                                 self.mean,
                                                                                 2) + (
                                                                                         other.n_samples - 1) * other.variance +
                                                                             other.n_samples * pow(
                    other.mean, 2) - (self.n_samples +
                                      other.n_samples) * pow(combined_mean, 2))
        return combined_variance

    def get(self):
        if self.n_samples == 1 or self.variance < 0:
            return 0
        else:
            return sqrt(self.variance)


class LengthOfDiagonal(IncrementalAttribute):

    def __init__(self, far_left, far_right, far_up, far_down):
        self.far_left = far_left
        self.far_right = far_right
        self.far_up = far_up
        self.far_down = far_down

    def __add__(self, other):
        far_left = other.far_left if other.far_left < self.far_left else \
            self.far_left
        far_right = other.far_right if other.far_right > self.far_right else \
            self.far_right
        far_up = other.far_up if other.far_up < self.far_up else \
            self.far_up
        far_down = other.far_down if other.far_down > self.far_down else \
            self.far_down

        return LengthOfDiagonal(far_left, far_right, far_up, far_down)

    def get(self):
        a = self.far_right - self.far_left + 1
        b = self.far_down - self.far_up + 1
        return sqrt(pow(a, 2) + pow(b, 2))


class FirstHuMoment(IncrementalAttribute):

    def __init__(self, pixel):
        if type(pixel) != list:
            self.pixels = [pixel]
        else:
            self.pixels = pixel

    def __add__(self, other):
        pixels = self.pixels + other.pixels
        return FirstHuMoment(pixels)

    def get(self):
        x_mass_center, y_mass_center = mass_center(self.pixels)
        moment00 = spatial_moments(self.pixels, 0, 0)
        moment20, moment02, moment11 = central_moments(self.pixels,
                                                       x_mass_center,
                                                       y_mass_center)
        norm_moment20, norm_moment02, norm_moment11 = \
            normalize_central_moments(moment20, moment02, moment11, moment00)
        return pow(norm_moment20 - norm_moment02, 2) + 4 * pow(norm_moment11, 2)
