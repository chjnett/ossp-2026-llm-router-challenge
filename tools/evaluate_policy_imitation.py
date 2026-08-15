# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""Train-only oracle 선택 모사의 OOF 일반화 가능성을 진단한다.

각 outer fold의 학습 부분에서만 실제 score/cost로 예산 제약 oracle 라벨을
만든다. 프롬프트 hashed ridge가 그 라벨을 맞히도록 학습한 뒤 T22 점수 예측과
혼합한다. held-out outcome은 최종 공식 채점에만 사용한다.

이 파일은 탐색 도구다. 복수 fold에서 재현될 때만 런타임 헤드로 승격한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from router.allocate import allocate
from router.config import Config, effective_util
from router.constants import MODEL_IDS, TIERS
from router.data import budget_multipliers, load_dataset
from router.harness import evaluate
from router.hash_features import extract_hash_features
from router.heads import _hash_ridge_apply, _hash_ridge_fit
from router.pipeline import Prediction, pick_all_tiers, predict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "t22-balanced-cap10.json"


def _one_hot(picks: np.ndarray) -> np.ndarray:
    target = np.zeros((len(picks), len(MODEL_IDS)), dtype=float)
    target[np.arange(len(picks)), picks] = 1.0
    return target


def _fit_imitation(
    config: Config,
    train,
    *,
    bins: int,
    alpha: float,
) -> dict[str, tuple]:
    design = extract_hash_features(train.texts, bins)
    multipliers = budget_multipliers(train.policy)
    util = effective_util(config, len(train), multipliers)
    fitted = {}
    for tier in TIERS:
        oracle = allocate(
            train.score,
            train.cost,
            multiplier=multipliers[tier],
            util=util[tier],
            keys=train.keys,
        ).picks
        fitted[tier] = _hash_ridge_fit(design, _one_hot(oracle), alpha)
    return fitted


def _apply_imitation(
    fitted: dict[str, tuple], texts, *, bins: int
) -> dict[str, np.ndarray]:
    design = extract_hash_features(texts, bins)
    return {
        tier: np.clip(_hash_ridge_apply(design, fitted[tier]), 0.0, 1.0)
        for tier in TIERS
    }


def _blend_prediction(
    base: Prediction,
    imitation: dict[str, np.ndarray],
    weight: float,
) -> Prediction:
    scores = {
        tier: np.clip(
            (1.0 - weight) * base.score_for_tier(tier)
            + weight * imitation[tier],
            0.0,
            1.0,
        )
        for tier in TIERS
    }
    return Prediction(
        s_hat=scores["fast"],
        c_hat=base.c_hat,
        sd=base.sd,
        allow=base.allow,
        versions=base.versions,
        s_hat_by_tier=scores,
        allow_by_tier=base.allow_by_tier,
        c_hat_by_tier=base.c_hat_by_tier,
        sd_by_tier=base.sd_by_tier,
    )


def _collect_oof(config: Config, dataset, *, k: int, bins: int, alpha: float):
    folds = dataset.folds(k)
    n = len(dataset)
    shape = (n, len(MODEL_IDS))
    score = {tier: np.zeros(shape) for tier in TIERS}
    imitation = {tier: np.zeros(shape) for tier in TIERS}
    cost = {tier: np.zeros(shape) for tier in TIERS}
    sd = {tier: np.zeros(shape) for tier in TIERS}
    allow = {tier: np.zeros(shape, dtype=bool) for tier in TIERS}
    versions = {}

    for held_out in folds:
        fit_idx = np.setdiff1d(np.arange(n), held_out, assume_unique=True)
        fit_part = dataset.subset(fit_idx)
        texts = tuple(dataset.texts[i] for i in held_out)
        base = predict(config, fit_part, texts)
        fitted = _fit_imitation(config, fit_part, bins=bins, alpha=alpha)
        im = _apply_imitation(fitted, texts, bins=bins)
        for tier in TIERS:
            score[tier][held_out] = base.score_for_tier(tier)
            imitation[tier][held_out] = im[tier]
            cost[tier][held_out] = base.cost_for_tier(tier)
            sd[tier][held_out] = base.sd_for_tier(tier)
            allow[tier][held_out] = base.allow_for_tier(tier)
        versions = base.versions

    base = Prediction(
        s_hat=score["fast"],
        c_hat=cost["fast"],
        sd=sd["fast"],
        allow=allow["fast"],
        versions=versions,
        s_hat_by_tier=score,
        allow_by_tier=allow,
        c_hat_by_tier=cost,
        sd_by_tier=sd,
    )
    return base, imitation


def _collect_dev(config: Config, train, dev, *, bins: int, alpha: float):
    base = predict(config, train, dev.texts)
    fitted = _fit_imitation(config, train, bins=bins, alpha=alpha)
    return base, _apply_imitation(fitted, dev.texts, bins=bins)


def _report(label: str, config: Config, dataset, base, imitation, weights) -> None:
    multipliers = budget_multipliers(dataset.policy)
    print(label)
    print("weight  final       fast       balanced   premium    ratios")
    for weight in weights:
        prediction = _blend_prediction(base, imitation, weight)
        picks = pick_all_tiers(config, prediction, dataset.keys, multipliers)
        result = evaluate(dataset, picks)
        tiers = result.tiers
        print(
            f"{weight:6.3f}  {float(result.final_score):.9f}  "
            f"{float(tiers['fast'].score):.6f}  "
            f"{float(tiers['balanced'].score):.6f}  "
            f"{float(tiers['premium'].score):.6f}  "
            + "/".join(f"{float(tiers[t].budget_ratio):.3f}" for t in TIERS)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.35),
    )
    parser.add_argument("--skip-dev", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    train = load_dataset("train")
    base, imitation = _collect_oof(
        config, train, k=args.folds, bins=args.bins, alpha=args.alpha
    )
    _report(f"Train OOF ({args.folds}-fold)", config, train, base, imitation, args.weights)

    if not args.skip_dev:
        dev = load_dataset("dev")
        base, imitation = _collect_dev(
            config, train, dev, bins=args.bins, alpha=args.alpha
        )
        _report("Dev (train fit)", config, dev, base, imitation, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
