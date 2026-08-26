# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Final low-risk knob sweep around t43 for a small CV/dev gain.

Sweeps the remaining untested knobs using the promotion metric (Train-CV)
as primary, dev as secondary, gate verified separately. Looking for any
point that Pareto-improves t43 (CV up, gate same/safer).
    PYTHONPATH=src .venv-data/bin/python experiments/probe_final_sweep.py
"""

from __future__ import annotations

import json
import os
import tempfile


from router.config import Config
from router.data import load_dataset
from router.pipeline import run_cv, run_on_split

ROOT = "/Users/cheonhyeonjun/skt_Routing/ossp-2026-llm-router-challenge"
T43 = f"{ROOT}/experiments/configs/t43-bal-h112.json"
BASE = json.load(open(T43))


def run_cfg(c, label):
    fd, fn = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    json.dump(c, open(fn, "w"))
    train = load_dataset("train")
    dev = load_dataset("dev")
    cfg = Config.load(fn)
    cv = run_cv(cfg, train, k=5)
    ev, _, _ = run_on_split(cfg, train, dev)
    os.unlink(fn)
    cvf = float(cv.final_score)
    devf = float(ev.final_score)
    pv = float(ev.tiers["premium"].budget_ratio)
    print(f"[{label:44s}] cv={cvf:.4f} dev={devf:.4f} prem_budget={pv:.4f}")


def main():
    train = load_dataset("train")
    dev = load_dataset("dev")

    # control t43
    cfg = Config.load(T43)
    cv0 = run_cv(cfg, train, k=5)
    ev0, _, _ = run_on_split(cfg, train, dev)
    print(f"[control t43            ] cv={float(cv0.final_score):.4f} dev={float(ev0.final_score):.4f}"
          f" prem_budget={float(ev0.tiers['premium'].budget_ratio):.4f}")

    trials = []
    # fast/balanced risk_quantile (currently 0.8)
    for q in [0.7, 0.75, 0.85, 0.9]:
        c = json.loads(json.dumps(BASE))
        for t in ["fast", "balanced"]:
            c["cost"]["heads"][t]["risk_quantile"] = q
        c["id"] = f"fb-riskq-{q}"
        trials.append((f"fast/bal riskq={q}       ", c))
    # score alpha for fast/bal (currently 32000)
    for a in [16000, 64000]:
        c = json.loads(json.dumps(BASE))
        for t in ["fast", "balanced"]:
            c["score"]["heads"][t]["alpha"] = a
        c["id"] = f"fb-alpha-{a}"
        trials.append((f"fast/bal score alpha={a:6d}", c))
    # score bins (currently 256)
    for b in [128, 512]:
        c = json.loads(json.dumps(BASE))
        for t in ["fast", "balanced"]:
            c["score"]["heads"][t]["bins"] = b
        c["id"] = f"fb-bins-{b}"
        trials.append((f"fast/bal score bins={b:4d}", c))
    # size_penalty (alloc)
    for sp in [{"fast": 0.0, "balanced": 0.0, "premium": 0.0},
               {"fast": 2.25, "balanced": 2.5, "premium": 3.0},
               {"fast": 3.0, "balanced": 3.0, "premium": 3.0}]:
        c = json.loads(json.dumps(BASE))
        c["alloc"]["size_penalty"] = sp
        c["id"] = "sp-" + "-".join(str(int(v)) for v in sp.values())
        trials.append((f"size_penalty={sp}", c))
    # relative_cost_cap premium
    for rc in [40, 100]:
        c = json.loads(json.dumps(BASE))
        c["alloc"]["relative_cost_cap"] = {"balanced": 10, "premium": rc}
        c["id"] = f"rcc-prem-{rc}"
        trials.append((f"relative_cost_cap prem={rc}", c))
    # premium family_blend weight
    for w in [0.5, 0.6, 0.7, 0.8, 0.9]:
        c = json.loads(json.dumps(BASE))
        c["score"]["heads"]["premium"]["weight"] = w
        c["id"] = f"prem-blend-w-{w}"
        trials.append((f"premium blend weight={w}", c))

    for label, c in trials:
        run_cfg(c, label)


if __name__ == "__main__":
    raise SystemExit(main())
