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


def conformal_outlier_threshold_supervised(
    scores_calib: np.ndarray,
    y_calib: np.ndarray,
    alpha: float,
) -> float:
    """
    We assume:
      - scores_calib: model scores on the calibration set
                      (higher => more likely fraud)
      - y_calib: 0/1 labels for the same calibration points
      - alpha: target false positive rate bound on LEGIT (y=0) class

    We keep only the scores of legitimate transactions (y=0) and apply
    the standard conformal quantile:
        qhat = k-th smallest score, k = ceil((n+1)(1-alpha))
    Decision rule: OUTLIER (fraud) iff score > qhat.
    """
    scores_calib = np.asarray(scores_calib).reshape(-1)
    y_calib = np.asarray(y_calib).reshape(-1).astype(int)

    # use only legit samples to control FPR on clean data
    scores_clean = scores_calib[y_calib == 0]

    if scores_clean.size == 0:
        raise ValueError("No legitimate (y=0) points in calibration set.")

    s = np.sort(scores_clean)
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
