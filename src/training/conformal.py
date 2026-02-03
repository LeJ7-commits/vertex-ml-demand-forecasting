from __future__ import annotations
import numpy as np

def conformal_interval(y_cal: np.ndarray, yhat_cal: np.ndarray, yhat_test: np.ndarray, alpha: float = 0.1):
    """
    Simple split conformal for regression (symmetric residuals).
    Returns (lower, upper) arrays for yhat_test.

    y_cal: true values on calibration window
    yhat_cal: predictions on calibration window
    yhat_test: predictions on test
    alpha: miscoverage level (0.1 -> ~90% intervals)
    """
    y_cal = np.asarray(y_cal)
    yhat_cal = np.asarray(yhat_cal)
    yhat_test = np.asarray(yhat_test)

    resid = np.abs(y_cal - yhat_cal)
    q = np.quantile(resid, 1 - alpha, method="higher")
    lower = yhat_test - q
    upper = yhat_test + q
    return lower, upper, float(q)