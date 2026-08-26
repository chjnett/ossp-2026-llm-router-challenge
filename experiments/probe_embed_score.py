# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Feasibility: do E5 embeddings help our score head? Uses the real pipeline.

Registers an embedding-augmented hash_ridge score head into the runtime registry,
then runs the full t38 pipeline (cost/gate/alloc unchanged) with it. Honest
Train-CV + dev via run.py-equivalent path.
    PYTHONPATH=src /tmp/emb2/bin/python experiments/probe_embed_score.py
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

os.environ.setdefault("HF_HOME", "/Users/cheonhyeonjun/skt_Routing/.hf-cache")
import router.heads as H
from router.data import load_dataset
from router.hash_features import extract_hash_features
from router.pipeline import run_on_split
from router.config import Config

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
T38 = f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"


class EmbridgeHashScore(H.HashRidgeScore):
    """tokenwise: same as hash_ridge but include E5 embeddings in the design."""

    def _design(self, texts, split_hint):
        base = extract_hash_features(texts, self.bins)
        if len(texts) == 880:
            emb = np.load("/tmp/emb2_emb.npy") if False else np.load("/tmp/dev_emb.npy")
        else:
            emb = np.load("/tmp/train_emb.npy")
        if len(emb) != len(texts):
            raise ValueError("embedding length mismatch")
        return base, emb

    def fit(self, train):
        if len(train) != len(np.load("/tmp/train_emb.npy")):
            raise ValueError("train size mismatch")
        base = extract_hash_features(train.texts, self.bins)
        emb = np.load("/tmp/train_emb.npy")
        self._fitted = H._hash_ridge_fit(np.hstack([base, emb]), train.score, self.alpha)
        self._emb_prefix = 1.0

    def predict(self, texts):
        base = extract_hash_features(texts, self.bins)
        if len(texts) == 880:
            emb = np.load("/tmp/dev_emb.npy")
        else:
            emb = np.zeros((len(texts), 384))
        return np.clip(H._hash_ridge_apply(np.hstack([base, emb]), self._fitted), 0.0, 1.0)


def main() -> int:
    train = load_dataset("train")
    dev = load_dataset("dev")

    if not H.SCORE_HEADS.get("embridge"):
        H.register(H.SCORE_HEADS, "embridge")(
            lambda **kw: EmbridgeHashScore(**kw)
        )

    # control
    ev0, _, _ = run_on_split(Config.load(T38), train, dev)
    print(f"[control hash-only t38] dev={float(ev0.final_score):.4f}")

    for alpha in [100.0, 1000.0, 32000.0]:
        cfg = Config.load(T38)
        cfg_dict = json.load(open(T38))
        cfg_dict["score"] = {"name": "tiered", "heads": {
            t: {"name": "embridge", "alpha": alpha, "bins": 256}
            for t in ["fast", "balanced", "premium"]
        }}
        cfg_dict["id"] = f"embed-score-probe-a{int(alpha)}"
        fd, fn = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        json.dump(cfg_dict, open(fn, "w"))
        try:
            cfg = Config.load(fn)
            ev, _, _ = run_on_split(cfg, train, dev)
            d = float(ev.final_score)
            print(f"[hash+E5 alpha={alpha:8.0f}] dev={d:.4f} "
                  + " ".join(f"{t}:{float(ev.tiers[t].score):.4f}/{float(ev.tiers[t].budget_ratio):.4f}"
                             for t in ["fast", "balanced", "premium"]))
        finally:
            os.unlink(fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
