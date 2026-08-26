# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Test regret-weighted score training (idea from Seosamo/LightGBM win-prob).

Seosamo weights each query's training by regret = best_score - 2nd_best_score,
concentrating capacity on queries where a wrong route loses score. Our heads
weigh by inverse-variance (generation count) or unweighted. This tests whether
regret-weighting our hash_ridge score head improves Train-CV/dev.

Low risk: same features/heads, only the training-objective weights change.
    PYTHONPATH=src .venv-data/bin/python experiments/probe_regret.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

import router.heads as H
from router.data import load_dataset
from router.hash_features import extract_hash_features
from router.pipeline import run_on_split
from router.config import Config

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
T38 = f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"


def regret_weights(score: np.ndarray, generations: np.ndarray) -> np.ndarray:
    """w(q) = best - 2nd best score among models (Seosamo §5.2 regret weight)."""
    s = np.asarray(score, dtype=float)
    order = np.sort(s, axis=1)
    best = order[:, -1]
    second = order[:, -2]
    reg = np.maximum(best - second, 0.0)
    # queries with a generation bonus? Seosamo also inflates by n. Keep pure.
    return reg


class RegretScore(H.HashRidgeScore):
    def __init__(self, alpha=100.0, bins=256, base_weight="regret"):
        super().__init__(alpha=alpha, bins=bins)
        self.base_weight = base_weight

    def fit(self, train):
        from router.heads import _hash_ridge_fit_weighted
        design = extract_hash_features(train.texts, self.bins)
        w = regret_weights(train.score, train.generations)
        if self.base_weight == "invvar":
            w = np.asarray(train.generations, dtype=float) * 1.0
        elif self.base_weight == "regret_gen":
            w = regret_weights(train.score, train.generations) * np.asarray(
                train.generations, dtype=float)
        elif self.base_weight == "untie":
            w = np.maximum(regret_weights(train.score, train.generations), 1e-3)
        # clip zeros to avoid dropping all-tie queries entirely
        self._fitted = _hash_ridge_fit_weighted(design, train.score, self.alpha, w)


def main() -> int:
    train = load_dataset("train")
    dev = load_dataset("dev")

    if not H.SCORE_HEADS.get("regret"):
        H.register(H.SCORE_HEADS, "regret")(lambda **kw: RegretScore(**kw))

    base = json.load(open(T38))
    # control
    ev0, _, _ = run_on_split(Config.load(T38), train, dev)
    print(f"[control hash-only t38] dev={float(ev0.final_score):.4f}")

    for mode in ["regret", "regret_gen", "untie"]:
        for alpha in [100.0, 32000.0]:
            c = json.loads(json.dumps(base))
            c["score"] = {"name": "tiered", "heads": {
                t: {"name": "regret", "alpha": alpha, "bins": 256,
                    "base_weight": mode} for t in ["fast", "balanced", "premium"]
            }}
            c["id"] = f"regret-{mode}-a{int(alpha)}"
            fd, fn = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            json.dump(c, open(fn, "w"))
            try:
                ev, _, _ = run_on_split(Config.load(fn), train, dev)
                print(f"[regret {mode:10s} a={alpha:7.0f}] dev={float(ev.final_score):.4f} | "
                      + " ".join(f"{t}:{float(ev.tiers[t].score):.3f}/{float(ev.tiers[t].budget_ratio):.3f}"
                                 for t in ["fast", "balanced", "premium"]))
            finally:
                os.unlink(fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
