# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Find a score blend that does NOT hurt Train-CV but improves dev (low-regret
generalization gain). Honest EV probe: we need BOTH, else it's dev-overfit.

Sweeps gentle ridge-rig blends and reports (cv, dev) so we can pick a point
on the CV-neutral frontier.
    PYTHONPATH=src .venv-data/bin/python experiments/probe_neutral_blend.py
"""

from __future__ import annotations

import json
import os
import tempfile

from router.config import Config
from router.data import load_dataset
from router.pipeline import run_cv, run_on_split

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
BASE = json.load(open(f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"))
T = ["fast", "balanced", "premium"]


def prim(t):
    if t == "premium":
        return {"name": "family_blend",
                "base": {"name": "hash_ridge", "alpha": 32000, "bins": 256},
                "challenger": {"name": "family_hash_ridge", "alpha": 1000,
                               "bins": 64, "active_families": ["sym_math", "code_io"],
                               "min_family": 40},
                "weight": 0.75, "active_families": ["sym_math", "code_io"]}
    return {"name": "hash_ridge", "alpha": 32000, "bins": 256}


def make(alpha, w, second="ridge"):
    c = json.loads(json.dumps(BASE))
    if second == "ridge":
        sec = {"name": "hash_ridge", "alpha": alpha, "bins": 256}
    elif second == "weighted":
        sec = {"name": "hash_ridge_weighted", "alpha": alpha, "bins": 256}
    elif second == "response":
        sec = {"name": "hash_response", "alpha": alpha, "bins": 128}
    else:
        raise ValueError(second)
    c["score"] = {"name": "tiered",
                  "heads": {t: {"name": "blend", "heads": [prim(t), sec],
                                "weights": [1 - w, w]} for t in T}}
    c["id"] = f"neutral-{second}-a{alpha}-w{w}"
    fd, fn = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(c, open(fn, "w"))
    return fn


def main():
    train = load_dataset("train")
    dev = load_dataset("dev")

    cv0 = run_cv(Config.load(f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"),
                 train, k=5)
    base_cv = float(cv0.final_score)
    ev0, _, _ = run_on_split(Config.load(f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"),
                             train, dev)
    base_dev = float(ev0.final_score)
    print(f"control: cv={base_cv:.4f} dev={base_dev:.4f}")

    # gentle blends: small second-head weight, several diversities
    trials = [
        ("ridge", 32000, 0.15), ("ridge", 32000, 0.20),
        ("weighted", 100, 0.20), ("weighted", 100, 0.30),
        ("response", 1000, 0.2), ("response", 1000, 0.3),
    ]
    for second, alpha, w in trials:
        fn = make(alpha, w, second)
        cfg = Config.load(fn)
        cv = run_cv(cfg, train, k=5)
        ev, _, _ = run_on_split(cfg, train, dev)
        os.unlink(fn)
        cv_s = float(cv.final_score)
        dev_s = float(ev.final_score)
        d_cv = cv_s - base_cv
        d_dev = dev_s - base_dev
        flag = "OK" if d_cv >= -0.0005 and d_dev > 0 else ""
        print(f"[{second:8s} a{alpha:5d} w={w:.2f}] cv={cv_s:.4f}({d_cv:+.4f}) "
              f"dev={dev_s:.4f}({d_dev:+.4f}) {flag}")


if __name__ == "__main__":
    raise SystemExit(main())
