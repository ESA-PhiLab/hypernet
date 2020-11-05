import tensorflow.keras.backend as K


def jaccard_index(smooth: float = 0.0000001):
    '''
    Jaccard index training loss for segmentation like tasks. 
    Default smoothness coefficient comes from Cloud-Net example.
    '''

    def loss(y_true, y_pred):
        intersection = K.sum(y_true * y_pred, axis=(1, 2))
        y_true_sum = K.sum(y_true, axis=(1, 2))
        y_pred_sum = K.sum(y_pred, axis=(1, 2))

        jaccard = (intersection + smooth) / (y_true_sum + y_pred_sum - intersection + smooth)
        return 1 - jaccard

    return loss


def dice_coef():
    '''
    Dice coefficient training loss for segmentation like tasks.
    Internally uses Jaccard index.
    '''
    ji = jaccard_index(smooth=0)

    def loss(y_true, y_pred):
        ji_score = ji(y_true, y_pred)
        return 2 * ji_score / (ji_score + 1)

    return loss


def jaccard_metric(y_true, y_pred):
    intersection = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=(1, 2))
    y_true_sum = K.sum(K.round(y_true), axis=(1, 2))
    y_pred_sum = K.sum(K.round(y_pred), axis=(1, 2))

    return intersection / (y_true_sum + y_pred_sum - intersection)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    ret = true_positives / (possible_positives + K.epsilon())
    return ret


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    ret = true_positives / (predicted_positives + K.epsilon())
    return ret


def specificity(y_true, y_pred):
    y_true_neg = 1 - y_true
    y_pred_neg = 1 - y_pred
    true_negatives = K.sum(K.round(K.clip(y_true_neg * y_pred_neg, 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(y_true_neg, 0, 1)))
    ret = true_negatives / (possible_negatives + K.epsilon())
    return ret


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec*rec) / (prec + rec + K.epsilon()))
