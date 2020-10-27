from tensorflow import reduce_sum

def jaccard_index(smooth: float = 0.0000001):
    '''
    Jaccard index training loss for segmentation like tasks. 
    Default smoothness coefficient comes from Cloud-Net example.
    '''

    def loss(y_true, y_pred):
        intersection = reduce_sum(y_true * y_pred, axis=(1, 2))
        y_true_sum = reduce_sum(y_true, axis=(1, 2))
        y_pred_sum = reduce_sum(y_pred, axis=(1, 2))
        
        jaccard = (intersection + smooth) / (y_true_sum + y_pred_sum - intersection + smooth)
        return 1 - jaccard

    return loss
