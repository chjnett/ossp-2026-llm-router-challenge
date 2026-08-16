# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""설정 하나 = 실험 하나.

새 아이디어가 코드 수정을 요구하면 계약이 잘못 그어진 것이다 (RULES D6).
설정은 6단계 구현 이름과 할당 하이퍼파라미터만 담는다.

    {"id": "exp-041",
     "score": "family",
     "cost": {"name": "family", "quantile": 0.75},
     "gate":  {"name": "k1_cost_cap", "percentile": 85},
     "alloc": {"mu": 0.4, "util": {"fast": 0.90, "balanced": 0.88, "premium": 0.85}}}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .allocate import allocate, allocate_chance, cap_relative_cost
from .config import DEFAULT_UTIL, Config, effective_util
from .data import MODEL_IDS, TIERS, Dataset, budget_multipliers
from .harness import Evaluation, evaluate
from .heads import build_cost_head, build_gate, build_score_head

__all__ = [
    "Config", "DEFAULT_UTIL", "effective_util",
    "Prediction", "predict", "pick_all_tiers", "run_on_split", "run_cv",
]

@dataclass(frozen=True)
class Prediction:
    s_hat: np.ndarray
    c_hat: np.ndarray
    sd: np.ndarray
    allow: np.ndarray
    versions: Dict[str, str]
    s_hat_by_tier: Dict[str, np.ndarray] | None = None
    allow_by_tier: Dict[str, np.ndarray] | None = None
    c_hat_by_tier: Dict[str, np.ndarray] | None = None
    sd_by_tier: Dict[str, np.ndarray] | None = None

    def score_for_tier(self, tier: str) -> np.ndarray:
        return (
            self.s_hat_by_tier[tier]
            if self.s_hat_by_tier is not None
            else self.s_hat
        )

    def allow_for_tier(self, tier: str) -> np.ndarray:
        return (
            self.allow_by_tier[tier]
            if self.allow_by_tier is not None
            else self.allow
        )

    def cost_for_tier(self, tier: str) -> np.ndarray:
        return (
            self.c_hat_by_tier[tier]
            if self.c_hat_by_tier is not None
            else self.c_hat
        )

    def sd_for_tier(self, tier: str) -> np.ndarray:
        return self.sd_by_tier[tier] if self.sd_by_tier is not None else self.sd


def predict(config: Config, train: Dataset, texts: Sequence[str]) -> Prediction:
    """Train에서 적합하고 주어진 프롬프트에 대해 예측한다.

    적합에 쓰는 자료와 예측 대상은 반드시 분리한다. Dev는 보정 전용이며
    계수 학습에 섞지 않는다 (RULES D2).
    """

    score_head = build_score_head(config.score)
    cost_head = build_cost_head(config.cost)
    gate = build_gate(config.gate)

    score_head.fit(train)
    cost_head.fit(train)
    gate.fit(train)

    predict_tier = getattr(score_head, "predict_tier", None)
    s_hat_by_tier = (
        {tier: predict_tier(texts, tier) for tier in TIERS}
        if callable(predict_tier)
        else None
    )
    s_hat = (
        s_hat_by_tier["fast"]
        if s_hat_by_tier is not None
        else score_head.predict(texts)
    )
    predict_cost_tier = getattr(cost_head, "predict_tier", None)
    cost_by_tier = (
        {tier: predict_cost_tier(texts, tier) for tier in TIERS}
        if callable(predict_cost_tier)
        else None
    )
    c_hat_by_tier = (
        {tier: cost_by_tier[tier][0] for tier in TIERS}
        if cost_by_tier is not None
        else None
    )
    sd_by_tier = (
        {tier: cost_by_tier[tier][1] for tier in TIERS}
        if cost_by_tier is not None
        else None
    )
    c_hat, sd = (
        (c_hat_by_tier["fast"], sd_by_tier["fast"])
        if c_hat_by_tier is not None and sd_by_tier is not None
        else cost_head.predict(texts)
    )
    allow_tier = getattr(gate, "allow_tier", None)
    allow_by_tier = (
        {
            tier: allow_tier(
                texts,
                s_hat_by_tier[tier],
                c_hat_by_tier[tier] if c_hat_by_tier is not None else c_hat,
                tier,
            )
            for tier in TIERS
        }
        if callable(allow_tier) and s_hat_by_tier is not None
        else (
            {
                tier: allow_tier(
                    texts,
                    s_hat,
                    c_hat_by_tier[tier] if c_hat_by_tier is not None else c_hat,
                    tier,
                )
                for tier in TIERS
            }
            if callable(allow_tier)
            else None
        )
    )
    allow = (
        allow_by_tier["fast"]
        if allow_by_tier is not None
        else gate.allow(texts, s_hat, c_hat)
    )

    return Prediction(
        s_hat=s_hat,
        c_hat=c_hat,
        sd=sd,
        allow=allow,
        versions={
            "score": score_head.version,
            "cost": cost_head.version,
            "gate": gate.version,
        },
        s_hat_by_tier=s_hat_by_tier,
        allow_by_tier=allow_by_tier,
        c_hat_by_tier=c_hat_by_tier,
        sd_by_tier=sd_by_tier,
    )


def pick_all_tiers(
    config: Config,
    prediction: Prediction,
    keys: Sequence[str],
    multipliers: Mapping[str, float],
) -> Dict[str, np.ndarray]:
    if config.epsilon is not None:
        return {
            tier: allocate_chance(
                prediction.score_for_tier(tier),
                prediction.cost_for_tier(tier),
                prediction.sd_for_tier(tier),
                multiplier=multipliers[tier],
                epsilon=config.epsilon,
                allow=cap_relative_cost(
                    prediction.allow_for_tier(tier),
                    prediction.cost_for_tier(tier),
                    config.relative_cost_cap_for_tier(tier),
                ),
                keys=list(keys),
            ).picks
            for tier in TIERS
        }

    util = effective_util(config, len(keys), multipliers)
    return {
        tier: allocate(
            prediction.score_for_tier(tier),
            prediction.cost_for_tier(tier),
            multiplier=multipliers[tier],
            util=util[tier],
            allow=cap_relative_cost(
                prediction.allow_for_tier(tier),
                prediction.cost_for_tier(tier),
                config.relative_cost_cap_for_tier(tier),
            ),
            sd=prediction.sd_for_tier(tier),
            mu=config.mu_for_tier(tier),
            keys=list(keys),
            k1_cap=config.k1_cap_for_tier(tier),
        ).picks
        for tier in TIERS
    }


def run_on_split(config: Config, train: Dataset, target: Dataset) -> tuple[Evaluation, Prediction, Dict[str, np.ndarray]]:
    """Train에서 적합하고 target split에서 공식 채점기로 평가한다."""

    prediction = predict(config, train, target.texts)
    multipliers = budget_multipliers(target.policy)
    picks = pick_all_tiers(config, prediction, target.keys, multipliers)
    return evaluate(target, picks), prediction, picks


def run_cv(config: Config, dataset: Dataset, *, k: int = 5) -> Evaluation:
    """out-of-fold 평가. 모델 선택은 이 값으로 한다.

    **예측만 fold로 나누고 배분은 전체에서 한 번 한다.** 평가에서는 입력
    전체가 한 배치로 들어오므로 배분도 그 크기에서 일어난다. fold마다 따로
    배분하면 배치가 1/k로 쪼개져 크기 감쇠가 예산을 먹고, k를 키울수록
    더 심해진다(k=10에서 fold 176문항, 감쇠 0.151). 실측으로 fold별 배분
    0.6309 vs 전체 배분 0.6343이었고, 그 편향이 헤드마다 다르게 작용해
    비교 자체를 왜곡했다.

    Dev 점수는 보정과 통과 여부 확인용이고, 어느 구성이 나은지는 Train CV로
    정한다. Dev에 맞춰 고르기 시작하면 Dev가 학습 자료가 된다 (RULES D2).
    """

    folds = dataset.folds(k)
    n = len(dataset)
    n_models = len(MODEL_IDS)
    s_hat = np.zeros((n, n_models))
    c_hat = np.zeros((n, n_models))
    sd = np.zeros((n, n_models))
    allow = np.zeros((n, n_models), dtype=bool)
    versions: Dict[str, str] = {}
    s_hat_by_tier: Dict[str, np.ndarray] | None = None
    allow_by_tier: Dict[str, np.ndarray] | None = None
    c_hat_by_tier: Dict[str, np.ndarray] | None = None
    sd_by_tier: Dict[str, np.ndarray] | None = None

    for f, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        fit_part = dataset.subset(train_idx)
        texts = tuple(dataset.texts[i] for i in test_idx)
        prediction = predict(config, fit_part, texts)
        s_hat[test_idx] = prediction.s_hat
        if prediction.s_hat_by_tier is not None:
            if s_hat_by_tier is None:
                s_hat_by_tier = {
                    tier: np.zeros((n, n_models)) for tier in TIERS
                }
            for tier in TIERS:
                s_hat_by_tier[tier][test_idx] = prediction.s_hat_by_tier[tier]
        c_hat[test_idx] = prediction.c_hat
        sd[test_idx] = prediction.sd
        if prediction.c_hat_by_tier is not None:
            if c_hat_by_tier is None:
                c_hat_by_tier = {
                    tier: np.zeros((n, n_models)) for tier in TIERS
                }
                sd_by_tier = {
                    tier: np.zeros((n, n_models)) for tier in TIERS
                }
            for tier in TIERS:
                c_hat_by_tier[tier][test_idx] = prediction.c_hat_by_tier[tier]
                sd_by_tier[tier][test_idx] = prediction.sd_by_tier[tier]
        allow[test_idx] = prediction.allow
        if prediction.allow_by_tier is not None:
            if allow_by_tier is None:
                allow_by_tier = {
                    tier: np.zeros((n, n_models), dtype=bool) for tier in TIERS
                }
            for tier in TIERS:
                allow_by_tier[tier][test_idx] = prediction.allow_by_tier[tier]
        versions = prediction.versions

    oof = Prediction(
        s_hat=s_hat,
        c_hat=c_hat,
        sd=sd,
        allow=allow,
        versions=versions,
        s_hat_by_tier=s_hat_by_tier,
        allow_by_tier=allow_by_tier,
        c_hat_by_tier=c_hat_by_tier,
        sd_by_tier=sd_by_tier,
    )
    picks = pick_all_tiers(
        config, oof, dataset.keys, budget_multipliers(dataset.policy)
    )
    return evaluate(dataset, picks)
