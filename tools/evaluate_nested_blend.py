# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""블렌드 비율 선택까지 Train 내부에 가둔 nested OOF 평가.

일반 5-fold CV에서 0.05가 가장 좋았다는 사실 자체가 Train에 맞춘 선택이다.
이 도구는 outer fold를 완전히 보류하고, 각 outer-train의 inner CV에서만 비율을
고른 뒤 outer-test 예측을 모아 한 번에 배분한다. Dev는 전혀 읽지 않는다.

    PYTHONPATH=src python3 tools/evaluate_nested_blend.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from router.data import MODEL_IDS, budget_multipliers, load_dataset
from router.harness import evaluate
from router.pipeline import Config, Prediction, pick_all_tiers, predict, run_cv


def config_for_weight(weight: float) -> Config:
    mixture = {
        "name": "family_mixture",
        "strength": 0.75,
        "active_families": ["sym_math", "code_io"],
    }
    if weight <= 0:
        score = mixture
    else:
        score = {
            "name": "blend",
            "heads": [
                mixture,
                {
                    "name": "template",
                    "scheme": "digits",
                    "prior": 1.0,
                    "strength": 1.0,
                    "active_families": ["sym_math"],
                },
            ],
            "weights": [1.0 - weight, weight],
        }
    return Config(
        id=f"nested-blend-w{weight:g}",
        score=score,
        cost={"name": "ridge", "z": 1.28, "z_light": -0.5, "smearing": True},
        gate="none",
        alloc={
            "headroom": {"fast": 0.175, "balanced": 0.252, "premium": 0.675},
            "size_penalty": 2.0,
            "mu": 1.0,
        },
    )


def nested_evaluate(outer_k: int, inner_k: int, weights: list[float]) -> dict:
    train = load_dataset("train")
    outer_folds = train.folds(outer_k)
    n, m = len(train), len(MODEL_IDS)
    nested_arrays = [np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))]
    nested_allow = np.zeros((n, m), dtype=bool)
    base_arrays = [np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))]
    base_allow = np.zeros((n, m), dtype=bool)
    selections = []
    base_config = config_for_weight(0.0)

    for fold, test_idx in enumerate(outer_folds):
        train_idx = np.concatenate(
            [outer_folds[g] for g in range(outer_k) if g != fold]
        )
        fit_part = train.subset(train_idx)
        texts = tuple(train.texts[i] for i in test_idx)

        candidates = []
        for weight in weights:
            config = config_for_weight(weight)
            score = float(run_cv(config, fit_part, k=inner_k).final_score)
            candidates.append((score, -weight, weight, config))
        # 동점이면 템플릿 비중이 낮은 쪽을 택한다.
        inner_score, _tie, selected_weight, selected = max(candidates)
        selections.append(
            {
                "outer_fold": fold,
                "weight": selected_weight,
                "inner_score": inner_score,
                "candidates": {
                    f"{weight:g}": score
                    for score, _tie, weight, _config in candidates
                },
            }
        )
        print(
            f"outer {fold}: weight={selected_weight:g} "
            f"inner={inner_score:.12f}",
            flush=True,
        )

        nested = predict(selected, fit_part, texts)
        base = predict(base_config, fit_part, texts)
        for target, source in zip(nested_arrays, (nested.s_hat, nested.c_hat, nested.sd)):
            target[test_idx] = source
        nested_allow[test_idx] = nested.allow
        for target, source in zip(base_arrays, (base.s_hat, base.c_hat, base.sd)):
            target[test_idx] = source
        base_allow[test_idx] = base.allow

    multipliers = budget_multipliers(train.policy)
    nested_prediction = Prediction(*nested_arrays, nested_allow, {})
    base_prediction = Prediction(*base_arrays, base_allow, {})
    nested_picks = pick_all_tiers(
        base_config, nested_prediction, train.keys, multipliers
    )
    base_picks = pick_all_tiers(base_config, base_prediction, train.keys, multipliers)
    nested_eval = evaluate(train, nested_picks)
    base_eval = evaluate(train, base_picks)
    return {
        "outer_folds": outer_k,
        "inner_folds": inner_k,
        "weights": weights,
        "selections": selections,
        "nested": nested_eval.as_record(),
        "fixed_t1": base_eval.as_record(),
        "delta": float(nested_eval.final_score - base_eval.final_score),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--weights", type=float, nargs="+", default=[0, 0.05, 0.1, 0.15])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = nested_evaluate(args.outer_folds, args.inner_folds, args.weights)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
