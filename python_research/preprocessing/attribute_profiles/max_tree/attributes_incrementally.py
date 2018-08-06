from math import sqrt
from ..utils.data_types import Moments


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
        return (
            ((self.n_samples * self.mean) + (other.n_samples * other.mean)) /
            (self.n_samples + other.n_samples)
        )

    def _combine_variance(self, other):
        combined_mean = self._combine_mean(other)
        combined_variance = (
            1. / (self.n_samples + other.n_samples - 1.0) *
            (
                (self.n_samples - 1) * self.variance +
                self.n_samples * pow(self.mean, 2) +
                (other.n_samples - 1) * other.variance +
                other.n_samples * pow(other.mean, 2) -
                (self.n_samples + other.n_samples) * pow(combined_mean, 2)
            )
        )

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
        far_left = other.far_left if other.far_left < self.far_left else self.far_left
        far_right = other.far_right if other.far_right > self.far_right else self.far_right
        far_up = other.far_up if other.far_up < self.far_up else self.far_up
        far_down = other.far_down if other.far_down > self.far_down else self.far_down

        return LengthOfDiagonal(far_left, far_right, far_up, far_down)

    def get(self):
        a = self.far_right - self.far_left + 1
        b = self.far_down - self.far_up + 1
        return sqrt(pow(a, 2) + pow(b, 2))


class FirstHuMoment(IncrementalAttribute):
    def __init__(self, pixel=None, moments=None):
        if moments is None and pixel is not None:
            gray_level = float(pixel.value)
            self.x_mass_center = float(pixel.x)
            self.y_mass_center = float(pixel.y)
            self.moment00 = gray_level
            self.moment10 = self.x_mass_center * gray_level
            self.moment01 = self.y_mass_center * gray_level
            self.moment20 = pow(self.x_mass_center, 2) * gray_level
            self.moment02 = pow(self.y_mass_center, 2) * gray_level
        else:
            self.moment00 = moments.moment00
            self.moment10 = moments.moment10
            self.moment01 = moments.moment01
            self.moment20 = moments.moment20
            self.moment02 = moments.moment02
            self.x_mass_center = moments.x_mass_center
            self.y_mass_center = moments.y_mass_center

    def _combine_mass_centers(self, other):
        x_mass_center_combined = (
            (self.x_mass_center * self.moment00 + other.x_mass_center * other.moment00) /
            (self.moment00 + other.moment00)
        )
        y_mass_center_combined = (
            (self.y_mass_center * self.moment00 + other.y_mass_center * other.moment00) /
            (self.moment00 + other.moment00)
        )

        return x_mass_center_combined, y_mass_center_combined

    def __add__(self, other):
        x_mass_center, y_mass_center = self._combine_mass_centers(other)
        moment00 = self.moment00 + other.moment00
        moment10 = self.moment10 + other.moment10
        moment01 = self.moment01 + other.moment01
        moment20 = self.moment20 + other.moment20
        moment02 = self.moment02 + other.moment02
        return FirstHuMoment(
            moments=Moments(
                moment00, moment10, moment01, moment20, moment02, x_mass_center, y_mass_center
            )
        )

    def _calculate_central_moments(self):
        central_moment20 = self.moment20 - self.x_mass_center * self.moment10
        central_moment02 = self.moment02 - self.y_mass_center * self.moment01
        denominator = pow(self.moment00, 2)
        if denominator == 0:
            return 0, 0
        central_moment20 = central_moment20 / denominator
        central_moment02 = central_moment02 / denominator
        return central_moment20, central_moment02

    def get(self):
        central_moment20, central_moment02 = self._calculate_central_moments()
        return central_moment20 + central_moment02
