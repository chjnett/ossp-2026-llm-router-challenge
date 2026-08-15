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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


from router.data import TIERS, budget_multipliers, combine_datasets, load_dataset
from router.features import family_codes
from router.pipeline import Config, predict, run_cv, run_on_split
from router.stress import gate_passed, run_gate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.jsonl"
CHAMPION = ROOT / "experiments" / "champion.json"

# RULES C4. 0/150은 rule of three로 실제 파산확률 95% 상한이 2%다. 증명이 아니다.
MIN_GATE_TRIALS = 2000


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
        gate_prediction = (
            predict(config, combine_datasets(train, dev), dev.texts)
            if args.final_refit_gate
            else prediction
        )
        gate_results = run_gate(
            dev,
            gate_prediction.s_hat_by_tier or gate_prediction.s_hat,
            gate_prediction.c_hat_by_tier or gate_prediction.c_hat,
            family=family_codes(dev.texts),
            util=config.util,
            multipliers=budget_multipliers(dev.policy),
            allow=gate_prediction.allow_by_tier or gate_prediction.allow,
            sd=gate_prediction.sd_by_tier or gate_prediction.sd,
            mu=config.mu,
            trials=args.trials,
            size_penalty=config.size_penalty,
            headroom=config.headroom,
            epsilon=config.epsilon,
            relative_cost_cap=config.relative_cost_cap,
        )
    gate_ok = bool(gate_results) and gate_passed(gate_results)

    record = {
        "id": config.id,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "folds": args.folds,
        "config": config.as_dict(),
        "versions": prediction.versions,
        "cv": evaluation_record(cv),
        "dev": evaluation_record(dev_eval),
        "gate": {
            "ran": bool(gate_results),
            "fit_split": (
                "public-train-dev" if args.final_refit_gate else "train"
            ) if gate_results else None,
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
    if int(record["gate"].get("trials", 0)) < MIN_GATE_TRIALS:
        # 시행이 모자란 통과는 통과가 아니다. 적은 N에서의 0회는
        # 파산확률이 낮다는 증거가 되지 못한다.
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


def do_adopt(args) -> int:
    """점수가 낮은 설정을 **일부러** 챔피언으로 세운다.

    자동 승격은 CV로만 판단한다. 그것이 맞다 — 점수가 낮은 설정이 조용히
    챔피언이 되면 안 된다. 다만 점수를 안전과 바꾸는 결정은 사람이 내리고,
    그럴 때 ``champion.json``을 손으로 고치면 **왜 그랬는지가 사라진다.**

    이 경로는 게이트 조건(N>=2000 통과, 세 등급 예산 통과)은 그대로 요구하고
    이유를 반드시 받아 기록에 남긴다.
    """

    records = [r for r in load_records() if r["config"]["id"] == args.config_id]
    if not records:
        print(f"오류: {args.config_id} 기록이 없다. 먼저 run으로 돌려야 한다", file=sys.stderr)
        return 2
    record = records[-1]

    if not record["gate"]["passed"]:
        print("오류: 게이트를 통과하지 못한 설정은 챔피언이 될 수 없다", file=sys.stderr)
        return 2
    if int(record["gate"].get("trials", 0)) < MIN_GATE_TRIALS:
        print(
            f"오류: 게이트 시행이 {record['gate'].get('trials', 0)}회다. "
            f"{MIN_GATE_TRIALS}회 이상이어야 한다",
            file=sys.stderr,
        )
        return 2
    if not record["cv"]["all_passed"]:
        print("오류: 교차검증에서 예산을 넘긴 등급이 있다", file=sys.stderr)
        return 2

    previous = None
    if CHAMPION.exists():
        previous = json.loads(CHAMPION.read_text(encoding="utf-8"))

    record = dict(record)
    record["adopted"] = {
        "reason": args.reason,
        "replaced": previous["config"]["id"] if previous else None,
        "cv_delta": (
            float(record["cv"]["final_score"])
            - float(previous["cv"]["final_score"]) if previous else None
        ),
    }
    CHAMPION.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    delta = record["adopted"]["cv_delta"]
    print(f"챔피언 교체: {record['adopted']['replaced']} → {args.config_id}")
    print(f"  CV {float(record['cv']['final_score']):.4f}" + (f" ({delta:+.4f})" if delta else ""))
    print(f"  Dev {float(record['dev']['final_score']):.4f}")
    print(f"  이유: {args.reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="설정 하나를 돌려 기록에 남긴다")
    p_run.add_argument("config", type=str)
    p_run.add_argument("--folds", type=int, default=5)
    p_run.add_argument("--trials", type=int, default=300)
    p_run.add_argument("--skip-gate", action="store_true")
    p_run.add_argument(
        "--final-refit-gate",
        action="store_true",
        help="실제 제출처럼 Train+Dev 재적합 모델을 파산 게이트로 검사한다",
    )
    p_run.set_defaults(func=do_run)

    p_board = sub.add_parser("leaderboard", help="기록을 CV 순으로 정렬해 보여준다")
    p_board.add_argument("--limit", type=int, default=30)
    p_board.set_defaults(func=do_leaderboard)

    p_champ = sub.add_parser("champion", help="현재 챔피언 설정을 보여준다")
    p_champ.set_defaults(func=do_champion)

    p_adopt = sub.add_parser(
        "adopt", help="점수가 낮은 설정을 일부러 챔피언으로 세운다 (이유 필수)"
    )
    p_adopt.add_argument("config_id", type=str)
    p_adopt.add_argument("--reason", type=str, required=True)
    p_adopt.set_defaults(func=do_adopt)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
