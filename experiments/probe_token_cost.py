# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""NEW: token-component cost head — predict output/input tokens, derive cost.

Our current cost head regresses total log-cost directly. Idea (Seosamo's
insight + physical structure): output_tokens is the dominant, heavy-tailed
cost driver (>90% of token mass, 4x weight; max 36k vs light median ~hundred).
Predict output_tokens with a log-quantile ridge + input_tokens with a simple
model, DERIVE cost = fixed + in*rate_in + out*rate_out, then apply the same
risk_quantile/unseen machinery.

Feasibility: OOF cost accuracy + full dev routing vs t43.
    PYTHONPATH=src .venv-data/bin/python experiments/probe_token_cost.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

import router.heads as H
from router.config import Config
from router.data import load_dataset
from router.hash_features import extract_hash_features
from router.data import cost_from_tokens
from router.pipeline import run_on_split

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
T43 = f"{ROOT}/experiments/configs/t43-bal-h112.json"


def log_quantile_fit(design, target, alpha, compute_sd=True):
    """Multi-output ridge on log-target with residual std."""
    logt = np.log(np.maximum(target, 1e-12))
    fitted = H._hash_ridge_fit(design, logt, alpha)
    pred = H._hash_ridge_apply(design, fitted)
    resid = logt - pred
    return fitted, resid.std(axis=0)


class TokenCost:
    """Cost head: predict output_tokens (log-quantile) + input_tokens, derive cost."""

    def __init__(self, bins=256, out_alpha=100.0, in_alpha=100.0,
                 risk_quantile=None, unseen_family_risk=False,
                 unseen_risk_boost=1.0, risk_oof_folds=None):
        self.bins = int(bins)
        self.out_alpha = float(out_alpha)
        self.in_alpha = float(in_alpha)
        self.risk_quantile = risk_quantile
        self.unseen_family_risk = bool(unseen_family_risk)
        self.unseen_risk_boost = float(unseen_risk_boost)
        self.risk_oof_folds = risk_oof_folds
        self.version = f"tokencost(o={out_alpha},i={in_alpha},rq={risk_quantile})"

    def fit(self, train):
        from router.features import family_codes, FAMILIES
        design = extract_hash_features(train.texts, self.bins)
        # --- predict OUTPUT tokens (dominant, heavy tail) via log-quantile
        self._out_fit, self._out_sd = log_quantile_fit(design, train.output_tokens, self.out_alpha)
        # --- predict INPUT tokens (near-deterministic) via simple ridge
        self._in_fit, self._in_sd = log_quantile_fit(design, train.input_tokens, self.in_alpha)
        self._policy = train.policy
        # risk factor on derived cost, mirroring HashRidgeCost
        out_pred = np.exp(H._hash_ridge_apply(design, self._out_fit))
        in_pred = np.exp(H._hash_ridge_apply(design, self._in_fit))
        base_cost = cost_from_tokens(in_pred, out_pred, train.policy)
        self._seen_families = np.bincount(
            family_codes(train.texts), minlength=len(FAMILIES)).astype(bool)
        ratio = train.cost / np.maximum(base_cost, 1e-12)
        quantile = self.risk_quantile if self.risk_quantile else 0.5
        codes = family_codes(train.texts)
        global_risk = np.quantile(ratio, quantile, axis=0)
        self._risk = np.tile(global_risk, (len(FAMILIES), 1))
        for f in range(len(FAMILIES)):
            mask = codes == f
            if mask.sum() >= 40:
                self._risk[f] = np.quantile(ratio[mask], quantile, axis=0)
        self._risk[:, 0] = 1.0
        self._risk[:, 1:] = np.clip(self._risk[:, 1:], 1.0, 20.0)
        self._unseen_risk = self._risk.max(axis=0) * self.unseen_risk_boost
        self._unseen_risk[0] = 1.0

    def predict(self, texts):
        from router.features import family_codes
        design = extract_hash_features(texts, self.bins)
        out_t = np.exp(H._hash_ridge_apply(design, self._out_fit))
        in_t = np.exp(H._hash_ridge_apply(design, self._in_fit))
        cost = cost_from_tokens(in_t, out_t, self._policy)
        if self.risk_quantile is not None:
            codes = family_codes(texts)
            risk = self._risk[codes].copy()
            if self.unseen_family_risk:
                risk[~self._seen_families[codes]] = self._unseen_risk
            cost = cost * risk
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1 + 1e-12))
        return cost, cost * 0.3


def main() -> int:
    train = load_dataset("train")
    dev = load_dataset("dev")

    if not H.COST_HEADS.get("token_cost"):
        H.register(H.COST_HEADS, "token_cost")(lambda **kw: TokenCost(**kw))

    base = json.load(open(T43))
    # control t43
    ev0, _, _ = run_on_split(Config.load(T43), train, dev)
    print(f"[control t43 cost-hash-ridge] dev={float(ev0.final_score):.4f} | "
          + " ".join(f"{t}:{float(ev0.tiers[t].budget_ratio):.4f}" for t in ["fast","balanced","premium"]))

    for rq in [0.8, 0.68, None]:
        c = json.loads(json.dumps(base))
        # replace cost with token_cost for all tiers
        c["cost"] = {"name": "tiered", "heads": {
            t: {"name": "token_cost", "bins": 256, "out_alpha": 100.0,
                "in_alpha": 100.0, "risk_quantile": rq,
                "unseen_family_risk": True, "unseen_risk_boost": 1.2 if rq else 1.0}
            for t in ["fast", "balanced", "premium"]}}
        c["id"] = f"tokencost-rq{rq}"
        fd, fn = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        json.dump(c, open(fn, "w"))
        try:
            ev, _, _ = run_on_split(Config.load(fn), train, dev)
            print(f"[token_cost rq={rq}        ] dev={float(ev.final_score):.4f} | "
                  + " ".join(f"{t}:{float(ev.tiers[t].score):.3f}/{float(ev.tiers[t].budget_ratio):.4f}"
                             for t in ["fast","balanced","premium"]))
        finally:
            os.unlink(fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
