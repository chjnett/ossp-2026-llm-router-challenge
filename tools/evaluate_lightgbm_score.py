# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""LightGBM 점수 헤드의 Train-only OOF 가치를 진단한다.

LightGBM은 오프라인 탐색에만 필요하다. 제출 런타임에는 포함하지 않으며, 복수
fold에서 이득이 재현될 때만 트리를 정적 JSON/NumPy 추론기로 옮긴다. 각 outer
fold의 held-out outcome은 적합이나 조기 종료에 사용하지 않는다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - 개발용 선택 의존성
    raise SystemExit("이 진단에는 오프라인 LightGBM 설치가 필요합니다") from exc

from router.config import Config
from router.constants import MODEL_IDS, TIERS
from router.data import budget_multipliers, load_dataset
from router.features import FAMILIES, extract, family_codes
from router.harness import evaluate
from router.hash_features import extract_hash_features
from router.pipeline import Prediction, pick_all_tiers, predict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "t26-balanced-code-cap35.json"


def _design(texts, feature_set: str, bins: int) -> np.ndarray:
    codes = family_codes(texts)
    one_hot = np.eye(len(FAMILIES), dtype=float)[codes]
    dense = np.column_stack([extract(texts), extract_hash_features(texts, 16)[:, :14]])
    if feature_set == "dense":
        return np.column_stack([dense, one_hot])
    hashed = extract_hash_features(texts, bins)
    if feature_set == "hash":
        return np.column_stack([hashed, one_hot])
    if feature_set == "both":
        return np.column_stack([dense, hashed[:, 14:], one_hot])
    raise ValueError(f"알 수 없는 feature_set: {feature_set!r}")


def _fit(train, args) -> list:
    design = _design(train.texts, args.feature_set, args.bins)
    params = {
        "objective": "regression_l2",
        "metric": "None",
        "learning_rate": args.learning_rate,
        "num_leaves": args.leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data,
        "lambda_l1": args.l1,
        "lambda_l2": args.l2,
        "min_gain_to_split": args.min_gain,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_bin": args.max_bin,
        "verbosity": -1,
        "num_threads": 1,
        "seed": 20260816,
        "deterministic": True,
        "force_col_wise": True,
    }
    return [
        lgb.train(
            params,
            lgb.Dataset(design, label=train.score[:, model], free_raw_data=False),
            num_boost_round=args.rounds,
        )
        for model in range(len(MODEL_IDS))
    ]


def _apply(models: list, texts, args) -> np.ndarray:
    design = _design(texts, args.feature_set, args.bins)
    return np.clip(
        np.column_stack([model.predict(design) for model in models]),
        0.0,
        1.0,
    )


def _empty(n: int) -> dict[str, np.ndarray]:
    shape = (n, len(MODEL_IDS))
    return {tier: np.zeros(shape) for tier in TIERS}


def _collect_oof(config: Config, dataset, args):
    n = len(dataset)
    base_score, tree_score = _empty(n), _empty(n)
    cost, sd = _empty(n), _empty(n)
    allow = {
        tier: np.zeros((n, len(MODEL_IDS)), dtype=bool) for tier in TIERS
    }
    versions = {}
    folds = dataset.folds(args.folds)
    for fold, held_out in enumerate(folds):
        fit_idx = np.concatenate(
            [folds[i] for i in range(args.folds) if i != fold]
        )
        fit_part = dataset.subset(fit_idx)
        texts = tuple(dataset.texts[i] for i in held_out)
        base = predict(config, fit_part, texts)
        tree = _apply(_fit(fit_part, args), texts, args)
        for tier in TIERS:
            base_score[tier][held_out] = base.score_for_tier(tier)
            tree_score[tier][held_out] = tree
            cost[tier][held_out] = base.cost_for_tier(tier)
            sd[tier][held_out] = base.sd_for_tier(tier)
            allow[tier][held_out] = base.allow_for_tier(tier)
        versions = base.versions
        print(f"fold {fold + 1}/{args.folds} fitted", flush=True)
    return _prediction(base_score, cost, sd, allow, versions), tree_score


def _collect_dev(config: Config, train, dev, args):
    base = predict(config, train, dev.texts)
    tree = _apply(_fit(train, args), dev.texts, args)
    return base, {tier: tree for tier in TIERS}


def _prediction(scores, cost, sd, allow, versions) -> Prediction:
    return Prediction(
        s_hat=scores["fast"],
        c_hat=cost["fast"],
        sd=sd["fast"],
        allow=allow["fast"],
        versions=versions,
        s_hat_by_tier=scores,
        allow_by_tier=allow,
        c_hat_by_tier=cost,
        sd_by_tier=sd,
    )


def _blend(base: Prediction, tree, weight: float) -> Prediction:
    score = {
        tier: np.clip(
            (1.0 - weight) * base.score_for_tier(tier) + weight * tree[tier],
            0.0,
            1.0,
        )
        for tier in TIERS
    }
    cost = {tier: base.cost_for_tier(tier) for tier in TIERS}
    sd = {tier: base.sd_for_tier(tier) for tier in TIERS}
    allow = {tier: base.allow_for_tier(tier) for tier in TIERS}
    return _prediction(score, cost, sd, allow, base.versions)


def _report(label, config, dataset, base, tree, weights) -> None:
    print(f"\n{label}")
    print("weight  final       fast       balanced   premium    ratios")
    multipliers = budget_multipliers(dataset.policy)
    for weight in weights:
        candidate = _blend(base, tree, weight)
        picks = pick_all_tiers(config, candidate, dataset.keys, multipliers)
        result = evaluate(dataset, picks)
        print(
            f"{weight:6.3f}  {float(result.final_score):.9f}  "
            + "  ".join(f"{float(result.tiers[t].score):.6f}" for t in TIERS)
            + "  "
            + "/".join(
                f"{float(result.tiers[t].budget_ratio):.3f}" for t in TIERS
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--feature-set", choices=("dense", "hash", "both"), default="dense")
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--leaves", type=int, default=7)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-data", type=int, default=60)
    parser.add_argument("--l1", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--feature-fraction", type=float, default=1.0)
    parser.add_argument("--max-bin", type=int, default=31)
    parser.add_argument("--weights", type=float, nargs="+", default=(0, 0.025, 0.05, 0.1, 0.2, 0.35))
    parser.add_argument("--skip-dev", action="store_true")
    args = parser.parse_args()

    config = Config.load(args.config)
    train = load_dataset("train")
    base, tree = _collect_oof(config, train, args)
    _report(f"Train OOF ({args.folds}-fold)", config, train, base, tree, args.weights)
    if not args.skip_dev:
        dev = load_dataset("dev")
        base, tree = _collect_dev(config, train, dev, args)
        _report("Dev (Train fit)", config, dev, base, tree, args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
