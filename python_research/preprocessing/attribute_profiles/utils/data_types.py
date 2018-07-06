from math import sqrt


class Pixel:

    def __init__(self, x, y, gray_level):
        self.x = x
        self.y = y
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


class StdDevIncrementally:

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
        if type(other) == StdDevIncrementally:
            variance = self._combine_variance(other)
            mean = self._combine_mean(other)
            n_samples = self.n_samples + other.n_samples
        return StdDevIncrementally(mean=mean, n_samples=n_samples,
                                   variance=variance)

    def _combine_mean(self, other):
        return ((self.n_samples * self.mean) + (other.n_samples *
                                                other.mean)) / \
               (self.n_samples + other.n_samples)

    def _combine_variance(self, other):
        combined_mean = self._combine_mean(other)
        combined_variance = 1./(self.n_samples + other.n_samples - 1.0) * ((
            self.n_samples - 1) * self.variance + self.n_samples *
            pow(self.mean, 2) + (other.n_samples - 1) * other.variance +
            other.n_samples * pow(other.mean, 2) - (self.n_samples +
            other.n_samples) * pow(combined_mean, 2))
        return combined_variance

    def get_std(self):
        return sqrt(self.variance)