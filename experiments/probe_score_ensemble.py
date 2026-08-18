# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Score-ensemble probe: thislifehea's ridge+GBM score blend idea adapted to T38.

They reported +0.0056 dev from averaging ridge and GBM for BOTH score and cost.
We already tested cost-only (no gain). Here we test the SCORE blend: our tuned
per-tier score head averaged with a diverse second head, gate/alloc unchanged.

Candidate second heads (submittable, numpy-only) + one GBM (probe only).
    PYTHONPATH=src /tmp/skt-gbm/bin/python experiments/probe_score_ensemble.py
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from router.config import Config
from router.data import load_dataset
from router.pipeline import run_on_split

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"


T38_BASE = f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"


def load_t38() -> dict:
    return json.load(open(T38_BASE))


def rebase_score(cfg_dict, score_spec):
    """Return a T38 config with a swapped top-level score spec."""
    c = json.loads(json.dumps(cfg_dict))
    c["score"] = score_spec
    return c


def blend_spec(primary, secondary, w=0.5):
    """blend head: primary (weight 1-w) + secondary (weight w)."""
    return {
        "name": "blend",
        "heads": [primary, secondary],
        "weights": [1.0 - w, w],
    }


def primary_spec(tier):
    """The tuned T38 per-tier score head spec."""
    if tier == "premium":
        return {
            "name": "family_blend",
            "base": {"name": "hash_ridge", "alpha": 32000, "bins": 256},
            "challenger": {
                "name": "family_hash_ridge", "alpha": 1000, "bins": 64,
                "active_families": ["sym_math", "code_io"], "min_family": 40,
            },
            "weight": 0.75,
            "active_families": ["sym_math", "code_io"],
        }
    return {"name": "hash_ridge", "alpha": 32000, "bins": 256}


def secondary_spec(name):
    if name == "hash_response":
        return {"name": "hash_response"}
    if name == "hash_ridge_weighted":
        return {"name": "hash_ridge_weighted", "alpha": 100, "bins": 256}
    if name == "family_useful":
        return {"name": "family_useful"}
    if name == "gbs_score":
        return {"name": "gbs_score"}  # only if registered; else skip
    if name.startswith("ridge_a"):
        alpha = float(name.split("_a")[1])
        return {"name": "hash_ridge", "alpha": alpha, "bins": 256}
    return None


def main() -> int:
    train = load_dataset("train")
    dev = load_dataset("dev")
    base = load_t38()
    print(f"train={len(train.texts)} dev={len(dev.texts)}")

    control = Config.load(T38_BASE)
    ev0, _, _ = run_on_split(control, train, dev)
    base_score = float(ev0.final_score)
    print(f"[control hash_ridge T38] dev={base_score:.4f}")
    for t in ev0.tiers:
        r = ev0.tiers[t]
        print(f"    {t:9s} score={r.score:.4f} budget={r.budget_ratio:.4f}")

    candidates = ["hash_response", "hash_ridge_weighted", "family_useful",
                  "ridge_a100", "ridge_a1000"]
    for sec in candidates:
        sspec = secondary_spec(sec)
        if sspec is None:
            continue
        # tiered blend: all three tiers blend primary+secondary (w=0.5)
        tiered = {
            "name": "tiered",
            "heads": {
                t: blend_spec(primary_spec(t), sspec, w=0.5) for t in ev0.tiers
            },
        }
        c = rebase_score(base, tiered)
        fd, fn = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        json.dump(c, open(fn, "w"))
        t0 = time.perf_counter()
        ev, _, _ = run_on_split(Config.load(fn), train, dev)
        os.unlink(fn)
        score = float(ev.final_score)
        delta = score - base_score
        mark = "▲" if delta > 1e-4 else ("▼" if delta < -1e-4 else "=")
        print(f"[score blend ~{sec:22s}] dev={score:.4f} Δ={delta:+.4f} {mark} "
              f"({time.perf_counter()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
