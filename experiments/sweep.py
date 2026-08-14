# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""설정 격자를 훑어 파산 게이트를 통과하는 구성을 찾는다.

    PYTHONPATH=src python3 experiments/sweep.py --trials 150

캐시된 배열 위에서 도는 부분(할당·게이트)이 대부분이라 격자를 넓게 잡아도
된다. 결과는 run.py와 같은 results.jsonl에 쌓이고 챔피언 승격 규칙도 같다.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router.data import TIERS, budget_multipliers, load_dataset  # noqa: E402
from router.features import family_codes  # noqa: E402
from router.pipeline import Config, run_cv, run_on_split  # noqa: E402
from router.stress import gate_passed, run_gate  # noqa: E402

import run as runner  # noqa: E402


def grid():
    """문항 단위 비용 회귀 · 상방 편향 · K1 꼬리 컷 · 사용률의 격자.

    편향은 light 대비 상대적으로 건다. 모든 모델을 똑같이 부풀리면 한도의
    분모까지 커져 오히려 위험해진다.
    """

    for z, z_light, cap, util in itertools.product(
        (0.67, 1.28),
        (0.0, -0.5),
        (None, 90.0, 75.0),
        (0.90, 0.85, 0.80),
    ):
        gate = (
            {"name": "family_roi", "min_roi": 1.0}
            if cap is None
            else {"name": "k1_cost_cap", "percentile": cap, "min_roi": 1.0}
        )
        cap_label = "none" if cap is None else f"p{cap:g}"
        yield Config(
            id=f"ridge-z{z:g}L{z_light:g}-{cap_label}-u{util:g}",
            score="family",
            cost={"name": "ridge", "z": z, "z_light": z_light},
            gate=gate,
            alloc={"util": util},
            note="문항 단위 비용 회귀 + 비대칭 상방 편향",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    train, dev = load_dataset("train"), load_dataset("dev")
    multipliers = budget_multipliers(dev.policy)
    family = family_codes(dev.texts)

    rows = []
    configs = list(grid())
    print(f"설정 {len(configs)}개 × 게이트 {args.trials}회\n")
    print(f"{'id':30s} {'CV':>7s} {'Dev':>7s} {'파산':>6s}  {'Dev 비율 f/b/p':>20s}")

    for config in configs:
        started = time.perf_counter()
        cv = run_cv(config, train, k=args.folds)
        dev_eval, prediction, _ = run_on_split(config, train, dev)
        results = run_gate(
            dev,
            prediction.s_hat_by_tier or prediction.s_hat,
            prediction.c_hat,
            family=family,
            util=config.util,
            multipliers=multipliers,
            allow=prediction.allow_by_tier or prediction.allow,
            sd=prediction.sd,
            mu=config.mu,
            trials=args.trials,
            relative_cost_cap=config.relative_cost_cap,
        )
        failures = sum(r.tiers[t].failures for r in results for t in TIERS)
        ok = gate_passed(results)

        record = {
            "id": config.id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": runner.git_sha(),
            "config": config.as_dict(),
            "versions": prediction.versions,
            "cv": cv.as_record(),
            "dev": dev_eval.as_record(),
            "gate": {
                "ran": True,
                "trials": args.trials,
                "passed": ok,
                "scenarios": {
                    r.scenario: {t: r.tiers[t].failures for t in TIERS}
                    for r in results
                },
            },
            "seconds": round(time.perf_counter() - started, 2),
        }
        with runner.RESULTS.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        promoted = runner.maybe_promote(record)
        rows.append((float(cv.final_score), config.id, failures, ok))

        ratios = " ".join(
            f"{float(dev_eval.tiers[t].budget_ratio):.2f}" for t in TIERS
        )
        mark = " ★승격" if promoted else (" 통과" if ok else "")
        print(
            f"{config.id:30s} {float(cv.final_score):7.4f} "
            f"{float(dev_eval.final_score):7.4f} {failures:6d}  {ratios:>20s}{mark}"
        )

    clean = [r for r in rows if r[3]]
    print()
    if clean:
        clean.sort(reverse=True)
        print(f"게이트 통과 {len(clean)}개. 최고 CV: {clean[0][1]} ({clean[0][0]:.4f})")
    else:
        print("게이트를 통과한 구성이 없다. 격자를 더 보수적인 쪽으로 넓혀야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
