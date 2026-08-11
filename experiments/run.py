# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""실험 러너와 챔피언 순위표.

    PYTHONPATH=src python3 experiments/run.py run experiments/configs/family.json
    PYTHONPATH=src python3 experiments/run.py leaderboard
    PYTHONPATH=src python3 experiments/run.py champion

승격 규칙(RULES D7): ① Train CV 가중 최종이 챔피언 초과 ② 파산 게이트 통과.
**둘 다** 만족할 때만 챔피언이 바뀐다. 그래서 실패한 실험의 비용이 0이 되고,
방향 전환이 판단이 아니라 순위표를 읽는 일이 된다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from router.data import TIERS, budget_multipliers, load_dataset
from router.features import family_codes
from router.pipeline import Config, pick_all_tiers, predict, run_cv, run_on_split
from router.stress import gate_passed, run_gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.jsonl"
CHAMPION = ROOT / "experiments" / "champion.json"


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def evaluation_record(evaluation) -> dict:
    return evaluation.as_record()


def do_run(args) -> int:
    config = Config.load(Path(args.config))
    train, dev = load_dataset("train"), load_dataset("dev")

    started = time.perf_counter()
    cv = run_cv(config, train, k=args.folds)
    dev_eval, prediction, _ = run_on_split(config, train, dev)

    gate_results = []
    if not args.skip_gate:
        gate_results = run_gate(
            dev,
            prediction.s_hat,
            prediction.c_hat,
            family=family_codes(dev.texts),
            util=config.util,
            multipliers=budget_multipliers(dev.policy),
            allow=prediction.allow,
            sd=prediction.sd,
            mu=config.mu,
            trials=args.trials,
        )
    gate_ok = bool(gate_results) and gate_passed(gate_results)

    record = {
        "id": config.id,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "config": config.as_dict(),
        "versions": prediction.versions,
        "cv": evaluation_record(cv),
        "dev": evaluation_record(dev_eval),
        "gate": {
            "ran": bool(gate_results),
            "trials": args.trials if gate_results else 0,
            "passed": gate_ok,
            "scenarios": {
                r.scenario: {t: r.tiers[t].failures for t in TIERS}
                for r in gate_results
            },
        },
        "seconds": round(time.perf_counter() - started, 2),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[{config.id}]  ({record['seconds']}s)")
    print("  Train CV")
    print(f"{cv}")
    print("  Dev")
    print(f"{dev_eval}")
    if gate_results:
        total = sum(r.tiers[t].failures for r in gate_results for t in TIERS)
        print(f"  파산 게이트: {'통과' if gate_ok else '불통과'} (총 {total}회 초과)")
        for r in gate_results:
            if not r.passed:
                print(f"    {r}")
    else:
        print("  파산 게이트: 건너뜀 (--skip-gate)")

    promoted = maybe_promote(record)
    print("  → 챔피언 승격" if promoted else "  → 챔피언 유지")
    return 0


def maybe_promote(record: dict) -> bool:
    """CV 가중 초과 **그리고** 게이트 통과일 때만 승격한다."""

    if not record["gate"]["passed"]:
        return False
    if not record["cv"]["all_passed"]:
        return False
    new_score = float(record["cv"]["final_score"])
    if CHAMPION.exists():
        current = json.loads(CHAMPION.read_text(encoding="utf-8"))
        if new_score <= float(current["cv"]["final_score"]):
            return False
    CHAMPION.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return True


def load_records() -> list[dict]:
    if not RESULTS.exists():
        return []
    rows = []
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def do_leaderboard(args) -> int:
    rows = load_records()
    if not rows:
        print("아직 기록이 없다. run 을 먼저 실행한다.")
        return 0
    rows.sort(key=lambda r: float(r["cv"]["final_score"]), reverse=True)
    champion_id = None
    if CHAMPION.exists():
        champion_id = json.loads(CHAMPION.read_text(encoding="utf-8"))["id"]

    print(f"{'':2s} {'id':22s} {'CV 가중':>9s} {'Dev 가중':>9s} {'게이트':>7s} "
          f"{'Dev fast/bal/prem 비율':>26s}")
    for row in rows[: args.limit]:
        mark = "★" if row["id"] == champion_id else " "
        gate = "통과" if row["gate"]["passed"] else ("불통과" if row["gate"]["ran"] else "생략")
        ratios = " ".join(
            f"{float(row['dev']['tiers'][t]['budget_ratio']):.3f}" for t in TIERS
        )
        print(
            f"{mark:2s} {row['id'][:22]:22s} "
            f"{float(row['cv']['final_score']):9.4f} "
            f"{float(row['dev']['final_score']):9.4f} {gate:>7s} {ratios:>26s}"
        )
    print("\n참고선: all-light 0.6193 · hash-regex(Dev) 0.6954 · oracle(Dev) 0.8037")
    return 0


def do_champion(args) -> int:
    if not CHAMPION.exists():
        print("아직 챔피언이 없다. 게이트를 통과한 구성이 나와야 한다.")
        return 1
    row = json.loads(CHAMPION.read_text(encoding="utf-8"))
    print(f"챔피언: {row['id']}  (commit {row['commit'][:7]}, {row['at']})")
    print(json.dumps(row["config"], ensure_ascii=False, indent=2))
    print(f"CV 가중  {float(row['cv']['final_score']):.4f}")
    print(f"Dev 가중 {float(row['dev']['final_score']):.4f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="설정 하나를 돌려 기록에 남긴다")
    p_run.add_argument("config", type=str)
    p_run.add_argument("--folds", type=int, default=5)
    p_run.add_argument("--trials", type=int, default=300)
    p_run.add_argument("--skip-gate", action="store_true")
    p_run.set_defaults(func=do_run)

    p_board = sub.add_parser("leaderboard", help="기록을 CV 순으로 정렬해 보여준다")
    p_board.add_argument("--limit", type=int, default=30)
    p_board.set_defaults(func=do_leaderboard)

    p_champ = sub.add_parser("champion", help="현재 챔피언 설정을 보여준다")
    p_champ.set_defaults(func=do_champion)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
