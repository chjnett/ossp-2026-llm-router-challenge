# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Cheap decision gate: does GBM predict log-cost better than ridge on Train OOF?

Runs beside the real pipeline (borrows features/target only). Nothing copied.
    PYTHONPATH=src /tmp/skt-gbm/bin/python experiments/probe_gbm_costgate.py
"""

from __future__ import annotations

import numpy as np

import router.heads as H
from router.data import load_dataset
from router.hash_features import extract_hash_features
from sklearn.ensemble import (HistGradientBoostingRegressor)


def oof_rmse(X, y, fit_fn, folds):
    pred = np.full_like(y, np.nan, dtype=float)
    for tr, te in folds:
        coef, arr = fit_fn(X[tr], y[tr])
        pred[te] = arr(X[te])
    return np.sqrt(np.mean((pred - y) ** 2)), pred


def main() -> int:
    train = load_dataset("train")
    design = extract_hash_features(train.texts, 256)
    log_cost = np.log(train.cost)
    folds = train.folds(5)
    codes = train.keys
    _ = codes
    print(f"design={design.shape} cost={train.cost.shape}")

    ridge_state = H._hash_ridge_fit(design, log_cost, alpha=100.0)
    ridge_pred = H._hash_ridge_apply(design, ridge_state)
    ridge_rmse = np.sqrt(np.mean((ridge_pred - log_cost) ** 2, axis=0))

    # GBM kfold OOF (manual, per model)
    for label, mk in [
        ("HistGBM", lambda rs: HistGradientBoostingRegressor(
            l2_regularization=1.0, max_leaf_nodes=15, min_samples_leaf=10,
            max_iter=100, learning_rate=0.1, random_state=rs)),
    ]:
        oof = np.full_like(log_cost, np.nan)
        for m in range(3):
            for te in folds:
                tr = np.setdiff1d(np.arange(len(design)), te)
                g = mk(m)
                g.fit(design[tr], log_cost[tr, m])
                oof[te, m] = g.predict(design[te])
        rmse = np.sqrt(np.mean((oof - log_cost) ** 2, axis=0))
        print(f"{label:9s}  OOF log-cost RMSE per model {rmse.round(4)}"

        "  total={:.4f}".format(np.sqrt(np.mean((oof - log_cost) ** 2))))

    # ridge in-sample for reference
    print(f"{'ridge':9s}  in-sample RMSE per model {ridge_rmse.round(4)}")
    # ridge OOF
    oof_ridge = H._hash_ridge_oof_apply(design, log_cost, folds, 100.0)
    print(f"{'ridge':9s}  OOF log-cost RMSE per model "
          f"{np.sqrt(np.mean((oof_ridge - log_cost) ** 2, axis=0)).round(4)}")

    # Ensemble (ridge OOF + HistGBM OOF) mean models
    h_of = np.full_like(log_cost, np.nan)
    for m in range(3):
        for te in folds:
            tr = np.setdiff1d(np.arange(len(design)), te)
            g = HistGradientBoostingRegressor(
                l2_regularization=1.0, max_leaf_nodes=15, min_samples_leaf=10,
                max_iter=100, learning_rate=0.1, random_state=m)
            g.fit(design[tr], log_cost[tr, m])
            h_of[te, m] = g.predict(design[te])
    for w in [0.3, 0.5, 0.7]:
        ens = w * h_of + (1 - w) * oof_ridge
        rmse = np.sqrt(np.mean((ens - log_cost) ** 2))
        print(f"ens w_gbm={w:.1f} OOF total RMSE={rmse:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
