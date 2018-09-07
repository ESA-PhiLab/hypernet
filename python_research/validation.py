from typing import NamedTuple
from keras.models import Model


class ValidationResult(NamedTuple):
    acc: float
    loss: float


def validate(model: Model, training_set):
    evaluation_result = model.evaluate(training_set.x_test, training_set.y_test)
    acc = evaluation_result[model.metrics_names.index('acc')]
    loss = evaluation_result[model.metrics_names.index('loss')]
    return ValidationResult(acc=acc, loss=loss)
