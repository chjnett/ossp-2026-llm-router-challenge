# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""파산 게이트를 사용률 후보별로 돌려 배포값을 정한다.

    PYTHONPATH=src python3 experiments/gate.py [--trials 2000]

RULES C4: 파산 0%가 처음 관측된 설정을 그대로 쓰지 않고 사용률을 한 단계
더 보수적으로 잡아 배포한다. N=300에서 0회는 rule of three로 실제 파산확률의
95% 상한이 약 1%다. 0%의 증명이 아니다.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from router.data import TIERS, cost_from_tokens, load_dataset
from router.features import FAMILIES, family_codes
from router.stress import gate_passed, run_gate

MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def family_predictions(train, dev):
    """계열 평균 예측. 지금 시점의 가장 단순한 정책이자 게이트 회귀 검사 대상."""

    fam_tr = family_codes(train.texts)
    fam_dev = family_codes(dev.texts)
    n_f, n_m = len(FAMILIES), 3
    score = np.zeros((n_f, n_m))
    log_in = np.zeros((n_f, n_m))
    log_out = np.zeros((n_f, n_m))
    for f in range(n_f):
        m = fam_tr == f
        if not m.any():
            m = np.ones(len(train), dtype=bool)
        score[f] = train.score[m].mean(axis=0)
        log_in[f] = np.log1p(train.input_tokens[m]).mean(axis=0)
        log_out[f] = np.log1p(train.output_tokens[m]).mean(axis=0)
    s_hat = score[fam_dev]
    c_hat = cost_from_tokens(
        np.expm1(log_in[fam_dev]), np.expm1(log_out[fam_dev]), dev.policy
    )
    return s_hat, c_hat, fam_dev


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train, dev = load_dataset("train"), load_dataset("dev")
    s_hat, c_hat, fam_dev = family_predictions(train, dev)

    print(f"파산 게이트 — 시나리오 6종 × {args.trials}회 × 3등급")
    print("배분은 예측 비용으로, 채점은 실제 비용으로 한다.\n")

    first_clean = None
    for util in (1.00, 0.95, 0.90, 0.85, 0.80, 0.75):
        t0 = time.perf_counter()
        results = run_gate(
            dev,
            s_hat,
            c_hat,
            family=fam_dev,
            util=util,
            multipliers=MULT,
            trials=args.trials,
            seed=args.seed,
        )
        worst = {
            t: max(r.tiers[t].failure_rate for r in results) for t in TIERS
        }
        total = sum(r.tiers[t].failures for r in results for t in TIERS)
        ok = gate_passed(results)
        if ok and first_clean is None:
            first_clean = util
        mark = "  ← 파산 0회" if ok else ""
        print(
            f"util={util:.2f}  총 파산 {total:6d}회  "
            f"최악 파산률 fast {worst['fast']:6.2%} "
            f"balanced {worst['balanced']:6.2%} premium {worst['premium']:6.2%}"
            f"  ({time.perf_counter() - t0:.1f}s){mark}"
        )
        if not ok:
            for r in results:
                if not r.passed:
                    print(f"    {r}")

    print()
    if first_clean is None:
        print("어떤 사용률에서도 파산 0회를 얻지 못했다. 비용 헤드부터 고쳐야 한다.")
        return 1
    deploy = round(first_clean - 0.03, 2)
    print(f"파산 0회 최초 관측: util={first_clean:.2f}")
    print(f"배포 권장값(RULES C4, 한 단계 더 보수적): util={deploy:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
