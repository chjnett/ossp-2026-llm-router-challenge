# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""후보 선택까지 Train 안에 가둔 nested OOF와 family holdout 평가.

Dev를 읽지 않는다. 첫 번째 설정은 고정 기준선이고, 나머지 설정을 포함한
후보군은 각 outer-train의 inner CV로만 선택한다. 별도로 한 계열을 통째로
학습에서 뺀 예측을 모아, 공개 계열 재가중 밖의 미지 분포에 대한 취약성을 잰다.

    PYTHONPATH=src python3 tools/evaluate_generalization.py \
      experiments/configs/t15-control-a100.json \
      experiments/configs/t15-shrunk-hash-a32000.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from router.config import Config
from router.data import MODEL_IDS, TIERS, Dataset, budget_multipliers, load_dataset
from router.features import FAMILIES, family_codes
from router.harness import evaluate
from router.pipeline import Prediction, pick_all_tiers, predict, run_cv


def _assert_compatible(configs: Sequence[Config]) -> None:
    """점수 헤드 외 제출 정책이 같아야 한 번의 nested 배분으로 비교할 수 있다."""

    if not configs:
        raise ValueError("설정이 하나 이상 필요하다")
    baseline = configs[0]
    for config in configs[1:]:
        if (config.cost, config.gate, config.alloc) != (
            baseline.cost,
            baseline.gate,
            baseline.alloc,
        ):
            raise ValueError(
                f"{config.id}: nested 후보는 score 외 cost/gate/alloc이 같아야 한다"
            )


def _empty_buffers(n: int) -> dict:
    shape = (n, len(MODEL_IDS))
    return {
        "score": {tier: np.zeros(shape) for tier in TIERS},
        "cost": {tier: np.zeros(shape) for tier in TIERS},
        "sd": {tier: np.zeros(shape) for tier in TIERS},
        "allow": {tier: np.zeros(shape, dtype=bool) for tier in TIERS},
    }


def _store(buffers: dict, index: np.ndarray, prediction: Prediction) -> None:
    for tier in TIERS:
        buffers["score"][tier][index] = prediction.score_for_tier(tier)
        buffers["cost"][tier][index] = prediction.cost_for_tier(tier)
        buffers["sd"][tier][index] = prediction.sd_for_tier(tier)
        buffers["allow"][tier][index] = prediction.allow_for_tier(tier)


def _as_prediction(buffers: dict) -> Prediction:
    return Prediction(
        s_hat=buffers["score"]["fast"],
        c_hat=buffers["cost"]["fast"],
        sd=buffers["sd"]["fast"],
        allow=buffers["allow"]["fast"],
        versions={},
        s_hat_by_tier=buffers["score"],
        allow_by_tier=buffers["allow"],
        c_hat_by_tier=buffers["cost"],
        sd_by_tier=buffers["sd"],
    )


def nested_evaluate(
    train: Dataset,
    configs: Sequence[Config],
    *,
    outer_k: int,
    inner_k: int,
) -> dict:
    _assert_compatible(configs)
    folds = train.folds(outer_k)
    selected_buffers = _empty_buffers(len(train))
    baseline_buffers = _empty_buffers(len(train))
    selections = []

    for fold, test_idx in enumerate(folds):
        train_idx = np.concatenate(
            [folds[other] for other in range(outer_k) if other != fold]
        )
        fit_part = train.subset(train_idx)
        texts = tuple(train.texts[i] for i in test_idx)
        candidates = []
        for order, config in enumerate(configs):
            score = float(run_cv(config, fit_part, k=inner_k).final_score)
            candidates.append((score, -order, config))
        inner_score, _tie, selected = max(candidates)
        selections.append(
            {
                "outer_fold": fold,
                "selected": selected.id,
                "inner_score": inner_score,
                "candidates": {
                    config.id: score for score, _order, config in candidates
                },
            }
        )
        print(
            f"outer {fold}: {selected.id} (inner={inner_score:.12f})",
            flush=True,
        )
        _store(selected_buffers, test_idx, predict(selected, fit_part, texts))
        _store(baseline_buffers, test_idx, predict(configs[0], fit_part, texts))

    multipliers = budget_multipliers(train.policy)
    selected_eval = evaluate(
        train,
        pick_all_tiers(
            configs[0], _as_prediction(selected_buffers), train.keys, multipliers
        ),
    )
    baseline_eval = evaluate(
        train,
        pick_all_tiers(
            configs[0], _as_prediction(baseline_buffers), train.keys, multipliers
        ),
    )
    return {
        "outer_folds": outer_k,
        "inner_folds": inner_k,
        "selections": selections,
        "selected": selected_eval.as_record(),
        "fixed_baseline": baseline_eval.as_record(),
        "delta": float(selected_eval.final_score - baseline_eval.final_score),
    }


def leave_one_family_out(train: Dataset, config: Config) -> dict:
    codes = family_codes(train.texts)
    buffers = _empty_buffers(len(train))
    multipliers = budget_multipliers(train.policy)
    per_family = {}

    for family_index, family in enumerate(FAMILIES):
        test_idx = np.where(codes == family_index)[0]
        if not len(test_idx):
            continue
        fit_idx = np.where(codes != family_index)[0]
        fit_part = train.subset(fit_idx)
        texts = tuple(train.texts[i] for i in test_idx)
        prediction = predict(config, fit_part, texts)
        _store(buffers, test_idx, prediction)
        picks = pick_all_tiers(
            config,
            prediction,
            tuple(train.keys[i] for i in test_idx),
            multipliers,
        )
        result = evaluate(train, picks, index=test_idx)
        per_family[family] = {"n": len(test_idx), **result.as_record()}
        print(
            f"LOFO {config.id:28s} {family:10s} n={len(test_idx):4d} "
            f"score={float(result.final_score):.6f} "
            f"passed={result.all_passed}",
            flush=True,
        )

    prediction = _as_prediction(buffers)
    overall = evaluate(
        train,
        pick_all_tiers(config, prediction, train.keys, multipliers),
    )
    return {"overall": overall.as_record(), "families": per_family}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", type=Path, nargs="+")
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-nested", action="store_true")
    parser.add_argument("--skip-lofo", action="store_true")
    args = parser.parse_args()

    configs = [Config.load(path) for path in args.configs]
    train = load_dataset("train")
    result = {"configs": [config.as_dict() for config in configs]}
    if not args.skip_nested:
        _assert_compatible(configs)
        result["nested"] = nested_evaluate(
            train,
            configs,
            outer_k=args.outer_folds,
            inner_k=args.inner_folds,
        )
    if not args.skip_lofo:
        result["leave_one_family_out"] = {
            config.id: leave_one_family_out(train, config) for config in configs
        }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
