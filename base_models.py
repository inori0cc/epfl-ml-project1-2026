"""Validated common entry points for the teammate's six course algorithms.

Inputs must already be imputed/scaled. Add an intercept column yourself when
required. Logistic methods require 0/1 labels; regression accepts real targets.
"""

import numpy as np
import implementations as core

METHODS = (
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
)
LOGISTIC = ("logistic_regression", "reg_logistic_regression")


def fit_model(
    method, y, tx, *, initial_w=None, max_iters=1000, gamma=0.01, lambda_=0.0, seed=None
):
    """Return (weights, unpenalized training loss) via implementations.py.

    Optional seed makes SGD reproducible and restores NumPy's caller RNG state.
    Closed-form methods ignore iteration options. Only regularized methods use
    lambda_. This API does not fit preprocessing or select a classification cutoff.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}; choose {METHODS}")
    y = np.asarray(y, dtype=float)
    tx = np.asarray(tx, dtype=float)
    if (
        y.ndim != 1
        or tx.ndim != 2
        or len(y) != len(tx)
        or not len(y)
        or tx.shape[1] == 0
    ):
        raise ValueError("Expected nonempty y (N,) and tx (N,D)")
    if not np.isfinite(y).all() or not np.isfinite(tx).all():
        raise ValueError("Impute missing values and remove infinities before fitting")
    if method in LOGISTIC and not np.isin(y, [0, 1]).all():
        raise ValueError("Logistic labels must be 0 or 1")
    if method in ("ridge_regression", "reg_logistic_regression") and (
        not np.isfinite(lambda_) or lambda_ < 0
    ):
        raise ValueError("lambda_ must be finite and nonnegative")
    fn = getattr(core, method)
    if method == "least_squares":
        return fn(y, tx)
    if method == "ridge_regression":
        return fn(y, tx, lambda_)
    w = (
        np.zeros(tx.shape[1])
        if initial_w is None
        else np.array(initial_w, dtype=float, copy=True)
    )
    if w.shape != (tx.shape[1],) or not np.isfinite(w).all():
        raise ValueError("Expected finite initial_w (D,)")
    if not isinstance(max_iters, (int, np.integer)) or max_iters < 0:
        raise ValueError("max_iters must be a nonnegative integer")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    if method == "reg_logistic_regression":
        return fn(y, tx, lambda_, w, max_iters, gamma)
    if method == "mean_squared_error_sgd" and seed is not None:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return fn(y, tx, w, max_iters, gamma)
        finally:
            np.random.set_state(state)
    return fn(y, tx, w, max_iters, gamma)


def predict_scores(method, tx, w):
    """Return linear scores or logistic probabilities; no automatic threshold."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(w, dtype=float)
    if tx.ndim != 2 or w.shape != (tx.shape[1],):
        raise ValueError("Expected tx (N,D), w (D,)")
    if not np.isfinite(tx).all() or not np.isfinite(w).all():
        raise ValueError("Inputs must be finite")
    scores = tx @ w
    return core.sigmoid(scores) if method in LOGISTIC else scores
