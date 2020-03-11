import time

import pytest

from ml_intuition.evaluation.time_metrics import timeit


class TestTimeMetrics:
    @pytest.mark.parametrize(
        'function',
        [
            (lambda: time.sleep(5)),
        ]
    )
    def test_timeit(self, function):
        function = timeit(function)
        _, result = function()
        assert type(result) == float, 'Assert the time type.'
        assert int(result) == 5, 'Assert the correct timing.'
