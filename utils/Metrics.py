import numpy as np


def computeMAP(class_fp_tps):
    aps = []
    for tp_fp in class_fp_tps.values():
        assert len(tp_fp['tp']) == len(tp_fp['fp']), "true positive and false positive must have the same length"
        percision, recall = compute_precision_recall(np.array(tp_fp['tp']), np.array(tp_fp['fp']), len(tp_fp['fp']))
        aps.append(compute_average_precision(percision, recall))
    
    return sum(aps) / len(aps)



def compute_precision_recall(tp, fp, n_positives):
    """ Compute Preision/Recall.
    Arguments:
        tp (np.array): true positives array.
        fp (np.array): false positives.
        n_positives (int): num positives.
    Returns:
        precision (np.array)
        recall (np.array)
    """
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / max(float(n_positives), 1)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return precision, recall

def compute_average_precision(precision, recall):
    """ Compute Avearage Precision by all points.
    Arguments:
        precision (np.array): precision values.
        recall (np.array): recall values.
    Returns:
        average_precision (np.array)
    """
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    ids = np.where(recall[1:] != recall[:-1])[0]
    average_precision = np.sum((recall[ids + 1] - recall[ids]) * precision[ids + 1])
    return average_precision