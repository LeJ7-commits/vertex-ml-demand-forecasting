import numpy as np

def conformal_quantile(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.1) -> float:
    """
    Split conformal for regression with absolute residuals.
    Returns q such that P(|Y - Ŷ| <= q) ≈ 1 - alpha on the calibration set.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = np.abs(y_true - y_pred)

    # Conformal quantile; using numpy quantile is fine for portfolio baseline.
    q = np.quantile(res, 1.0 - alpha)
    return float(q)

def conformal_interval(y_pred: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    y_pred = np.asarray(y_pred, dtype=float)
    lo = y_pred - q
    hi = y_pred + q
    return lo, hi

def interval_coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))
