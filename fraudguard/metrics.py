import numpy as np
from sklearn.metrics import precision_recall_curve

def find_best_threshold_by_f1(y_true, y_scores):
    """
    Exactly what you did:
    - precision_recall_curve gives precision/recall/thresholds
    - compute F1 for each threshold
    - pick argmax

    y_scores should be "fraud probability" for supervised, or any increasing score.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_scores = np.asarray(y_scores).reshape(-1)

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # thresholds has length len(precision)-1
    # F1 computed for all precision/recall entries:
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)

    # best idx in full array; thresholds needs idx-1 alignment sometimes
    best_idx = int(np.argmax(f1_scores))

    # if best_idx is last element, it doesn't have a matching threshold
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1

    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1, precision, recall, thresholds
