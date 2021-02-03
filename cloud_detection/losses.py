""" Custom loss functions and metrics for segmentation. """

import tensorflow.keras.backend as K
import numpy as np

# Import all other metrics used by the model
# to enable testing it.
from tensorflow.keras.metrics import binary_crossentropy, binary_accuracy


def Jaccard_index_loss(smooth: float = 0.0000001):
    '''
    Jaccard index training loss for segmentation like tasks. 
    Default smoothness coefficient comes from Cloud-Net example.
    '''

    def jaccard_index_loss(y_true: np.ndarray, y_pred: np.ndarray):
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        y_true_sum = K.sum(y_true, axis=(1, 2))
        y_pred_sum = K.sum(y_pred, axis=(1, 2))

        jaccard = (intersection + smooth) / (y_true_sum + y_pred_sum - intersection + smooth)
        return 1 - jaccard

    return jaccard_index_loss


def Jaccard_index_metric(smooth: float = 0.0000001):

    def jaccard_index_metric(y_true: np.ndarray, y_pred: np.ndarray):
        intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2))
        y_true_sum = K.sum(K.round(y_true), axis=(1, 2))
        y_pred_sum = K.sum(K.round(y_pred), axis=(1, 2))

        return (intersection + smooth) / (y_true_sum + y_pred_sum - intersection + smooth)

    return jaccard_index_metric


def Dice_coef_metric():
    '''
    Dice coefficient training loss for segmentation like tasks.
    Internally uses Jaccard index.
    '''
    ji = Jaccard_index_metric()

    def dice_coeff_metric(y_true: np.ndarray, y_pred: np.ndarray):
        ji_score = ji(y_true, y_pred)
        return 2 * ji_score / (ji_score + 1)

    return dice_coeff_metric


def recall(y_true: np.ndarray, y_pred: np.ndarray):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    ret = true_positives / (possible_positives + K.epsilon())
    return ret


def precision(y_true: np.ndarray, y_pred: np.ndarray):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ret = true_positives / (predicted_positives + K.epsilon())
    return ret


def specificity(y_true: np.ndarray, y_pred: np.ndarray):
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = K.sum(K.round(K.clip(y_true_neg * y_pred_neg, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true_neg, 0, 1)))
    ret = true_negatives / (possible_negatives + K.epsilon())
    return ret


# Same as Dice_coef_metric()
def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec*rec) / (prec + rec + K.epsilon()))
