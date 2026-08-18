# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the bankruptcy gate for a config using the exact run.py final-refit
recipe (predict fit on train+dev, scenarios, trials/scenario), returning overs.
Used to gate-check pareto candidates that improve Train-CV.
    PYTHONPATH=src .venv-data/bin/python experiments/probe_gate.py <config.json> [trials]
"""

from __future__ import annotations

import json
import sys

from router.config import Config
from router.data import TIERS, budget_multipliers, combine_datasets, load_dataset
from router.features import family_codes
from router.pipeline import predict
from router.stress import gate_passed, run_gate


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    config_fn = argv[0]
    trials = int(argv[1]) if len(argv) > 1 else 2000
    cfg = Config.load(config_fn)
    train = load_dataset("train")
    dev = load_dataset("dev")
    pred = predict(cfg, combine_datasets(train, dev), dev.texts)
    res = run_gate(
        dev,
        pred.s_hat_by_tier or pred.s_hat,
        pred.c_hat_by_tier or pred.c_hat,
        family=family_codes(dev.texts),
        util=cfg.util,
        multipliers=budget_multipliers(dev.policy),
        allow=pred.allow_by_tier or pred.allow,
        sd=pred.sd_by_tier or pred.sd,
        mu=cfg.mu,
        trials=trials,
        size_penalty=cfg.size_penalty,
        headroom=cfg.headroom,
        epsilon=cfg.epsilon,
        relative_cost_cap=cfg.relative_cost_cap,
    )
    total = sum(r.tiers[t].failures for r in res for t in TIERS)
    passed = gate_passed(res)
    print(f"{config_fn}: gate {'PASS' if passed else 'FAIL'}  overs={total}/{len(res)*trials}")
    for r in res:
        print(f"  {r.scenario:24s} " + " ".join(
            f"{t}={r.tiers[t].failures}" for t in TIERS))
    print(json.dumps({
        "config": cfg.id, "passed": bool(passed), "total_overs": total,
        "trials": trials,
    }, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
