# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""두 라우터의 공개 Dev 선택 차이를 실제 outcome으로 분해한다.

이 도구는 오프라인 진단 전용이다. 공개 Dev의 ID와 outcome을 사용하지만,
런타임 라우터나 학습 특징에는 전달하지 않는다.

예시:
    PYTHONPATH=src python3 tools/analyze_selection_gap.py \
      --candidate build/champ --reference build/hr \
      --output build/selection-gap.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from router.constants import MODEL_IDS, TIERS
from router.data import load_dataset, tier_weights
from router.features import family_of


def _decisions(folder: Path, tier: str) -> dict[str, str]:
    payload = json.loads((folder / f"{tier}.json").read_text(encoding="utf-8"))
    return {row["episode_id"]: row["model_id"] for row in payload["decisions"]}


def _blank() -> dict[str, float | int | dict[str, int]]:
    return {
        "episodes": 0,
        "different": 0,
        "reference_quality_gain_sum": 0.0,
        "reference_cost_delta_sum": 0.0,
        "candidate_better": 0,
        "reference_better": 0,
        "quality_tie": 0,
        "transitions": {},
    }


def _add(bucket: dict, transition: str, score_delta: float, cost_delta: float) -> None:
    bucket["episodes"] += 1
    bucket["different"] += 1
    bucket["reference_quality_gain_sum"] += score_delta
    bucket["reference_cost_delta_sum"] += cost_delta
    if score_delta > 1e-12:
        bucket["reference_better"] += 1
    elif score_delta < -1e-12:
        bucket["candidate_better"] += 1
    else:
        bucket["quality_tie"] += 1
    transitions = bucket["transitions"]
    transitions[transition] = transitions.get(transition, 0) + 1


def analyze(candidate: Path, reference: Path) -> dict:
    dev = load_dataset("dev")
    model_index = {model: i for i, model in enumerate(MODEL_IDS)}
    weights = {tier: float(weight) for tier, weight in tier_weights(dev.policy).items()}
    families = sorted({family_of(text) for text in dev.texts})

    by_family = {family: _blank() for family in families}
    by_tier = {tier: _blank() for tier in TIERS}
    weighted_gap = 0.0
    different = 0

    for tier in TIERS:
        candidate_pick = _decisions(candidate, tier)
        reference_pick = _decisions(reference, tier)
        for i, episode_id in enumerate(dev.episode_ids):
            c_model = candidate_pick[episode_id]
            r_model = reference_pick[episode_id]
            if c_model == r_model:
                continue
            different += 1
            c_idx = model_index[c_model]
            r_idx = model_index[r_model]
            score_delta = float(dev.score[i, r_idx] - dev.score[i, c_idx])
            cost_delta = float(dev.cost[i, r_idx] - dev.cost[i, c_idx])
            weighted_gap += weights[tier] * score_delta / len(dev)
            transition = f"{c_model}->{r_model}"
            family = family_of(dev.texts[i])
            _add(by_family[family], transition, score_delta, cost_delta)
            _add(by_tier[tier], transition, score_delta, cost_delta)

    for table in (by_family, by_tier):
        for row in table.values():
            row["reference_quality_gain_sum"] = round(
                float(row["reference_quality_gain_sum"]), 6
            )
            row["reference_cost_delta_sum"] = round(
                float(row["reference_cost_delta_sum"]), 9
            )
            row["weighted_quality_gap"] = 0.0

    # 위 공통 정리 루프에서 family 가중치를 계산하면 결정을 반복해서 읽게 된다.
    # 실제 가중 격차는 두 번째 단순 순회로 명시적으로 계산한다.
    family_weighted = defaultdict(float)
    tier_weighted = defaultdict(float)
    decisions = {
        (kind, tier): _decisions(folder, tier)
        for kind, folder in (("candidate", candidate), ("reference", reference))
        for tier in TIERS
    }
    for tier in TIERS:
        for i, episode_id in enumerate(dev.episode_ids):
            c_model = decisions[("candidate", tier)][episode_id]
            r_model = decisions[("reference", tier)][episode_id]
            delta = float(
                dev.score[i, model_index[r_model]] - dev.score[i, model_index[c_model]]
            )
            contribution = weights[tier] * delta / len(dev)
            family_weighted[family_of(dev.texts[i])] += contribution
            tier_weighted[tier] += contribution
    for family, value in family_weighted.items():
        by_family[family]["weighted_quality_gap"] = round(value, 9)
    for tier, value in tier_weighted.items():
        by_tier[tier]["weighted_quality_gap"] = round(value, 9)

    ranked_families = dict(
        sorted(
            by_family.items(),
            key=lambda item: item[1]["weighted_quality_gap"],
            reverse=True,
        )
    )
    return {
        "candidate": str(candidate),
        "reference": str(reference),
        "split": "dev",
        "episodes": len(dev),
        "tier_decisions": len(dev) * len(TIERS),
        "different_decisions": different,
        "reference_minus_candidate_weighted_score": round(weighted_gap, 9),
        "by_tier": by_tier,
        "by_family": ranked_families,
    }


def print_report(report: dict) -> None:
    print("# Selection gap: reference − candidate")
    print(f"different decisions: {report['different_decisions']} / "
          f"{report['tier_decisions']}")
    print(f"weighted quality gap: "
          f"{report['reference_minus_candidate_weighted_score']:+.6f}\n")
    print("## By tier")
    print("tier        diff   ref better   cand better   tie   weighted gap")
    for tier, row in report["by_tier"].items():
        print(f"{tier:10s} {row['different']:5d} {row['reference_better']:12d} "
              f"{row['candidate_better']:13d} {row['quality_tie']:5d} "
              f"{row['weighted_quality_gap']:+.6f}")
    print("\n## By family")
    print("family       diff   ref better   cand better   tie   weighted gap")
    for family, row in report["by_family"].items():
        print(f"{family:11s} {row['different']:5d} {row['reference_better']:12d} "
              f"{row['candidate_better']:13d} {row['quality_tie']:5d} "
              f"{row['weighted_quality_gap']:+.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.candidate, args.reference)
    print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
