import time

import pytest

from ml_intuition.evaluation.time_metrics import timeit


class TestTimeMetrics:
    @pytest.mark.parametrize(
        'function',
        [
            (lambda: time.sleep(1)),
        ]
    )
    def test_timeit(self, function):
        function = timeit(function)
        _, result = function()
        assert int(result) == 1, 'Assert the correct timing.'
