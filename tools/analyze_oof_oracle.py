# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""Train OOF 선택을 같은 실제 비용의 완전정보 oracle과 비교한다.

Dev를 읽지 않는다. 후보 라우터의 OOF 선택과 실제 지출 비율을 먼저 구한 뒤,
같은 실제 비용으로 true score/cost를 아는 oracle이 고를 수 있는 배분을 만든다.
둘의 점수 차이는 headroom 차이가 아니라 순위·모델선택 오류다. 공식 tier 예산을
전부 쓴 oracle도 별도로 계산해 안전마진으로 남겨 둔 상한을 분리한다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from router.allocate import allocate
from router.config import Config
from router.data import MODEL_IDS, TIERS, budget_multipliers, load_dataset, tier_weights
from router.features import FAMILIES, family_codes
from router.pipeline import Prediction, pick_all_tiers, predict


def oof_prediction(config: Config, folds_count: int) -> tuple:
    dataset = load_dataset("train")
    folds = dataset.folds(folds_count)
    shape = (len(dataset), len(MODEL_IDS))
    score = {tier: np.zeros(shape) for tier in TIERS}
    cost = {tier: np.zeros(shape) for tier in TIERS}
    sd = {tier: np.zeros(shape) for tier in TIERS}
    allow = {tier: np.zeros(shape, dtype=bool) for tier in TIERS}

    for fold, test_index in enumerate(folds):
        train_index = np.concatenate(
            [folds[other] for other in range(folds_count) if other != fold]
        )
        fit_part = dataset.subset(train_index)
        texts = tuple(dataset.texts[i] for i in test_index)
        fitted = predict(config, fit_part, texts)
        for tier in TIERS:
            score[tier][test_index] = fitted.score_for_tier(tier)
            cost[tier][test_index] = fitted.cost_for_tier(tier)
            sd[tier][test_index] = fitted.sd_for_tier(tier)
            allow[tier][test_index] = fitted.allow_for_tier(tier)

    prediction = Prediction(
        s_hat=score["fast"],
        c_hat=cost["fast"],
        sd=sd["fast"],
        allow=allow["fast"],
        versions={},
        s_hat_by_tier=score,
        allow_by_tier=allow,
        c_hat_by_tier=cost,
        sd_by_tier=sd,
    )
    return dataset, prediction


def _score(dataset, picks: np.ndarray) -> float:
    return float(dataset.score[np.arange(len(dataset)), picks].mean())


def _ratio(dataset, picks: np.ndarray) -> float:
    used = float(dataset.cost[np.arange(len(dataset)), picks].sum())
    return used / dataset.light_baseline_cost


def analyze(config: Config, folds_count: int) -> dict:
    dataset, prediction = oof_prediction(config, folds_count)
    multipliers = budget_multipliers(dataset.policy)
    weights = {tier: float(value) for tier, value in tier_weights(dataset.policy).items()}
    current = pick_all_tiers(config, prediction, dataset.keys, multipliers)
    codes = family_codes(dataset.texts)
    model_names = np.asarray(MODEL_IDS)
    result = {
        "config": config.as_dict(),
        "folds": folds_count,
        "tiers": {},
        "weighted": {
            "current_score": 0.0,
            "same_cost_oracle_score": 0.0,
            "full_budget_oracle_score": 0.0,
        },
    }

    for tier in TIERS:
        current_picks = current[tier]
        current_score = _score(dataset, current_picks)
        current_ratio = _ratio(dataset, current_picks)
        same_plan = allocate(
            dataset.score,
            dataset.cost,
            multiplier=current_ratio,
            util=1.0,
            keys=dataset.keys,
        )
        full_plan = allocate(
            dataset.score,
            dataset.cost,
            multiplier=multipliers[tier],
            util=1.0,
            keys=dataset.keys,
        )
        same_score = _score(dataset, same_plan.picks)
        full_score = _score(dataset, full_plan.picks)
        same_ratio = _ratio(dataset, same_plan.picks)
        full_ratio = _ratio(dataset, full_plan.picks)

        by_family = {}
        for code, family in enumerate(FAMILIES):
            mask = codes == code
            if not mask.any():
                continue
            delta = (
                dataset.score[mask, same_plan.picks[mask]]
                - dataset.score[mask, current_picks[mask]]
            )
            transitions = Counter(
                f"{model_names[a]}->{model_names[b]}"
                for a, b in zip(
                    current_picks[mask], same_plan.picks[mask], strict=True
                )
                if a != b
            )
            by_family[family] = {
                "n": int(mask.sum()),
                "same_cost_score_gain_sum": float(delta.sum()),
                "weighted_score_gain": float(weights[tier] * delta.sum() / len(dataset)),
                "different": int(np.count_nonzero(current_picks[mask] != same_plan.picks[mask])),
                "transitions": dict(transitions.most_common()),
            }
        by_family = dict(
            sorted(
                by_family.items(),
                key=lambda item: item[1]["weighted_score_gain"],
                reverse=True,
            )
        )
        result["tiers"][tier] = {
            "current_score": current_score,
            "current_ratio": current_ratio,
            "same_cost_oracle_score": same_score,
            "same_cost_oracle_ratio": same_ratio,
            "routing_regret": same_score - current_score,
            "full_budget_oracle_score": full_score,
            "full_budget_oracle_ratio": full_ratio,
            "unused_budget_regret": full_score - same_score,
            "by_family": by_family,
        }
        weight = weights[tier]
        result["weighted"]["current_score"] += weight * current_score
        result["weighted"]["same_cost_oracle_score"] += weight * same_score
        result["weighted"]["full_budget_oracle_score"] += weight * full_score

    weighted = result["weighted"]
    weighted["routing_regret"] = (
        weighted["same_cost_oracle_score"] - weighted["current_score"]
    )
    weighted["unused_budget_regret"] = (
        weighted["full_budget_oracle_score"]
        - weighted["same_cost_oracle_score"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(Config.load(args.config), args.folds)
    weighted = report["weighted"]
    print(
        "weighted: "
        f"current={weighted['current_score']:.6f} "
        f"same-cost-oracle={weighted['same_cost_oracle_score']:.6f} "
        f"routing-regret={weighted['routing_regret']:.6f} "
        f"unused-budget-regret={weighted['unused_budget_regret']:.6f}"
    )
    for tier in TIERS:
        row = report["tiers"][tier]
        print(
            f"{tier:9s} current={row['current_score']:.6f}/{row['current_ratio']:.3f} "
            f"same={row['same_cost_oracle_score']:.6f}/{row['same_cost_oracle_ratio']:.3f} "
            f"full={row['full_budget_oracle_score']:.6f}/{row['full_budget_oracle_ratio']:.3f}"
        )
        for family, family_row in list(row["by_family"].items())[:4]:
            print(
                f"  {family:10s} weighted_gain={family_row['weighted_score_gain']:+.6f} "
                f"different={family_row['different']}"
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
