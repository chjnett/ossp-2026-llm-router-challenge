# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""고정 예측 위에서 Balanced/Premium headroom만 빠르게 비교한다.

후보 점수는 Train OOF로만 고른다. Dev는 방향 확인용으로 함께 출력하며, 최선
후보는 별도 최종 재적합 스트레스 게이트를 통과해야 한다.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from router.config import Config
from router.constants import MODEL_IDS, TIERS
from router.data import budget_multipliers, combine_datasets, load_dataset
from router.features import family_codes
from router.harness import evaluate
from router.pipeline import Prediction, pick_all_tiers, predict
from router.stress import run_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "t26-balanced-code-cap35.json"


def _empty(n: int, *, boolean: bool = False) -> dict[str, np.ndarray]:
    shape = (n, len(MODEL_IDS))
    return {
        tier: np.zeros(shape, dtype=bool if boolean else float) for tier in TIERS
    }


def _oof(config: Config, dataset, folds: int) -> Prediction:
    n = len(dataset)
    score, cost, sd = _empty(n), _empty(n), _empty(n)
    allow = _empty(n, boolean=True)
    versions = {}
    split = dataset.folds(folds)
    for fold, held_out in enumerate(split):
        fit_idx = np.concatenate([split[i] for i in range(folds) if i != fold])
        fit_part = dataset.subset(fit_idx)
        texts = tuple(dataset.texts[i] for i in held_out)
        row = predict(config, fit_part, texts)
        for tier in TIERS:
            score[tier][held_out] = row.score_for_tier(tier)
            cost[tier][held_out] = row.cost_for_tier(tier)
            sd[tier][held_out] = row.sd_for_tier(tier)
            allow[tier][held_out] = row.allow_for_tier(tier)
        versions = row.versions
    return _prediction(score, cost, sd, allow, versions)


def _prediction(score, cost, sd, allow, versions) -> Prediction:
    return Prediction(
        score["fast"],
        cost["fast"],
        sd["fast"],
        allow["fast"],
        versions,
        score,
        allow,
        cost,
        sd,
    )


def _config(base: Config, balanced: float, premium: float) -> Config:
    alloc = dict(base.alloc)
    alloc["headroom"] = {
        "fast": float(base.headroom["fast"]),
        "balanced": balanced,
        "premium": premium,
    }
    return replace(base, id=f"b{balanced:g}-p{premium:g}", alloc=alloc)


def _row(config, dataset, prediction):
    picks = pick_all_tiers(
        config,
        prediction,
        dataset.keys,
        budget_multipliers(dataset.policy),
    )
    return evaluate(dataset, picks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--balanced", nargs="+", type=float, default=(0.9, 1.0, 1.1))
    parser.add_argument("--premium", nargs="+", type=float, default=(0.9, 1.0, 1.1, 1.2, 1.4))
    parser.add_argument("--stress-trials", type=int, default=0)
    parser.add_argument("--stress-top", type=int, default=4)
    args = parser.parse_args()

    base = Config.load(args.config)
    train, dev = load_dataset("train"), load_dataset("dev")
    oof = _oof(base, train, args.folds)
    dev_prediction = predict(base, train, dev.texts)
    rows = []
    for balanced in args.balanced:
        for premium in args.premium:
            config = _config(base, balanced, premium)
            cv = _row(config, train, oof)
            held_out = _row(config, dev, dev_prediction)
            rows.append((float(cv.final_score), balanced, premium, cv, held_out))
    rows.sort(reverse=True, key=lambda row: row[0])
    print("CV        Dev       B/P       CV ratios             Dev ratios")
    for score, balanced, premium, cv, held_out in rows:
        cv_ratio = "/".join(f"{float(cv.tiers[t].budget_ratio):.3f}" for t in TIERS)
        dev_ratio = "/".join(
            f"{float(held_out.tiers[t].budget_ratio):.3f}" for t in TIERS
        )
        print(
            f"{score:.9f} {float(held_out.final_score):.9f} "
            f"{balanced:.2f}/{premium:.2f}  {cv_ratio:21s} {dev_ratio}"
        )
    if args.stress_trials:
        final_fit = combine_datasets(train, dev)
        final_prediction = predict(base, final_fit, dev.texts)
        print("\nfinal-refit stress: failures and max ratio/limit")
        for _score, balanced, premium, _cv, _held_out in rows[: args.stress_top]:
            config = _config(base, balanced, premium)
            results = run_gate(
                dev,
                final_prediction.s_hat_by_tier or final_prediction.s_hat,
                final_prediction.c_hat_by_tier or final_prediction.c_hat,
                family=family_codes(dev.texts),
                util=config.util,
                multipliers=budget_multipliers(dev.policy),
                allow=final_prediction.allow_by_tier or final_prediction.allow,
                sd=final_prediction.sd_by_tier or final_prediction.sd,
                mu=config.mu,
                trials=args.stress_trials,
                size_penalty=config.size_penalty,
                headroom=config.headroom,
                epsilon=config.epsilon,
                relative_cost_cap=config.relative_cost_cap,
            )
            failures = {
                tier: sum(result.tiers[tier].failures for result in results)
                for tier in TIERS
            }
            maximum = {
                tier: max(result.tiers[tier].quantiles()[3] for result in results)
                for tier in TIERS
            }
            print(
                f"B/P={balanced:.2f}/{premium:.2f} failures={failures} "
                f"max={{{', '.join(f'{t}: {maximum[t]:.3f}' for t in TIERS)}}}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
