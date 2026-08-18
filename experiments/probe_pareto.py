# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Pareto probe: sweep safety/alloc knobs around T38, report (Train-CV, dev,
premium budget, premium stress) so the user can pick a defensible point.

CV = promotion signal. dev = hidden-set signal (we do NOT tune on it, but
we report it). Gate is evaluated separately by stress probe later.
    PYTHONPATH=src .venv-data/bin/python experiments/probe_pareto.py
"""

from __future__ import annotations

import json
import os
import tempfile

from router.config import Config
from router.data import load_dataset
from router.pipeline import run_cv, run_on_split

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
BASE_CFG = f"{ROOT}/experiments/configs/t38-prem-q068-urb120.json"


def mut(overrides_alloc=None, overrides_riskq=None, gate=None):
    """Return a T38-derived config dict with overrides."""
    c = json.load(open(BASE_CFG))
    if overrides_alloc:
        c["alloc"] = {**c["alloc"], **overrides_alloc}
    if overrides_riskq:
        for tier, q in overrides_riskq.items():
            c["cost"]["heads"][tier]["risk_quantile"] = q
    if gate:
        c["gate"] = {**c["gate"], **gate}
    c["id"] = "pareto-probe"
    return c


def run(c, label):
    fd, fn = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(c, open(fn, "w"))
    cfg = Config.load(fn)
    train = load_dataset("train")
    dev = load_dataset("dev")
    cv = run_cv(cfg, train, k=5)
    ev, _, _ = run_on_split(cfg, train, dev)
    os.unlink(fn)
    cvf = float(cv.final_score)
    devf = float(ev.final_score)
    pv = float(ev.tiers["premium"].budget_ratio)
    prem = float(ev.tiers["premium"].score)
    return cvf, devf, pv, prem


def main():
    base = mut()
    # control
    cvf, devf, pv, prem = run(base, "control")
    print(f"{'[control T38]':38s} cv={cvf:.4f} dev={devf:.4f} prem_budget={pv:.4f} prem_score={prem:.4f}")

    trials = []
    # premium risk_quantile sweep (champion has premium q=0.68, fast/bal=0.8)
    for q in [0.60, 0.65, 0.68, 0.72, 0.75]:
        c = mut(overrides_riskq={"premium": q})
        trials.append((f"prem riskq={q}       ", c))
    # headroom sweep (higher h = spend more)
    for h in [0.95, 1.0, 1.05, 1.12]:
        c = mut(overrides_alloc={"headroom": {"fast": 0.80, "balanced": h, "premium": 1.0}})
        trials.append((f"bal headroom={h}     ", c))
    for hf in [0.70, 0.75, 0.85]:
        c = mut(overrides_alloc={"headroom": {"fast": hf, "balanced": 1.075, "premium": 1.0}})
        trials.append((f"fast headroom={hf}   ", c))
    for hp in [0.95, 1.02, 1.06]:
        c = mut(overrides_alloc={"headroom": {"fast": 0.75, "balanced": 1.075, "premium": hp}})
        trials.append((f"prem headroom={hp}   ", c))
    # mu sweep (risk aversion on score std)
    for mu in [0.0, 0.05, 0.10]:
        c = mut(overrides_alloc={"mu": mu})
        trials.append((f"mu={mu}              ", c))

    for label, c in trials:
        cvf, devf, pv, prem = run(c, label)
        print(f"[{label}] cv={cvf:.4f} dev={devf:.4f} prem_budget={pv:.4f} prem_score={prem:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
