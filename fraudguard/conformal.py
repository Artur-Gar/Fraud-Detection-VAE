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

def conformal_pvalues_upper_tail(scores_clean: np.ndarray, scores_x: np.ndarray) -> np.ndarray:
    """
    Upper-tail conformal p-values:
    p(x) = (1 + #{i: s_i >= s(x)}) / (n + 1)
    Outlier at level alpha <=> p(x) <= alpha
    """
    s = np.asarray(scores_clean).reshape(-1)
    t = np.asarray(scores_x).reshape(-1)
    n = len(s)

    ge = (s[None, :] >= t[:, None]).sum(axis=1)
    return (1.0 + ge) / (n + 1.0)

def confusion_from_scores(y_true: np.ndarray, scores: np.ndarray, tau: float):
    """
    Generic confusion + rates for any score where larger => more 'fraud/outlier'.
    Rule: predict 1 if score > tau else 0.

    Returns dict with tn/fp/fn/tp and fpr/tpr/precision.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores).reshape(-1)
    y_pred = (scores > tau).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    fpr = fp / (fp + tn + 1e-12)
    tpr = tp / (tp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)

    return {
        "tau": float(tau),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": float(fpr),
        "tpr": float(tpr),
        "precision": float(precision),
    }
