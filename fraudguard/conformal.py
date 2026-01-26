import math
import numpy as np

def conformal_outlier_threshold(scores_clean: np.ndarray, alpha: float) -> float:
    """
    Threshold for outlier detection using the standard conformal quantile:
    qhat = k-th smallest score where k = ceil((n+1)(1-alpha))
    Decision rule: outlier iff score > qhat

    scores_clean: calibration scores from CLEAN data (legit only)
    alpha: target false positive rate bound on legit (e.g., 0.01 => 1%)
    """
    s = np.sort(np.asarray(scores_clean).reshape(-1))
    n = len(s)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])

def confusion_from_scores(y_true: np.ndarray, scores: np.ndarray, tau: float):
    y_pred = (scores > tau).astype(int)  # 1 = flagged as outlier/fraud
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    return fpr, tpr, tn, fp, fn, tp
