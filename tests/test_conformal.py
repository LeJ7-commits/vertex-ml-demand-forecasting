import numpy as np
from src.training.conformal import conformal_interval

def test_conformal_shapes():
    rng = np.random.default_rng(0)
    y_cal = rng.normal(size=100)
    yhat_cal = y_cal + rng.normal(scale=0.1, size=100)
    yhat_test = rng.normal(size=10)

    lo, hi, q = conformal_interval(y_cal, yhat_cal, yhat_test, alpha=0.1)
    assert lo.shape == (10,)
    assert hi.shape == (10,)
    assert q >= 0.0
    assert np.all(hi >= lo)