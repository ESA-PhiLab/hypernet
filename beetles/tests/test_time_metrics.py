import time

import pytest

from ml_intuition.evaluation.time_metrics import timeit


class TestTimeMetrics:
    @pytest.mark.parametrize(
        'function',
        [
            (lambda x: x**2)
        ]
    )
    def test_timeit_signe_return_value(self, function):
        function = timeit(function)
        result, t = function(2)
        assert result == 4
        assert isinstance(t, float)

    @pytest.mark.parametrize(
        'function',
        [
            (lambda x: (x, x**2, x ** 3))
        ]
    )
    def test_timeit_multiple_return_values(self, function):
        function = timeit(function)
        result, t = function(2)
        (x1, x2, x3), t = function(2)
        assert sum(result) == sum((x1, x2, x3)) == 14
        assert isinstance(t, float)
