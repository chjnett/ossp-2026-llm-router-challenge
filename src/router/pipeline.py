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

from .allocate import allocate, allocate_chance
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

    s_hat = score_head.predict(texts)
    c_hat, sd = cost_head.predict(texts)
    allow = gate.allow(texts, s_hat, c_hat)

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
                prediction.s_hat,
                prediction.c_hat,
                prediction.sd,
                multiplier=multipliers[tier],
                epsilon=config.epsilon,
                allow=prediction.allow,
                keys=list(keys),
            ).picks
            for tier in TIERS
        }

    util = effective_util(config, len(keys), multipliers)
    return {
        tier: allocate(
            prediction.s_hat,
            prediction.c_hat,
            multiplier=multipliers[tier],
            util=util[tier],
            allow=prediction.allow,
            sd=prediction.sd,
            mu=config.mu,
            keys=list(keys),
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

    for f, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        fit_part = dataset.subset(train_idx)
        texts = tuple(dataset.texts[i] for i in test_idx)
        prediction = predict(config, fit_part, texts)
        s_hat[test_idx] = prediction.s_hat
        c_hat[test_idx] = prediction.c_hat
        sd[test_idx] = prediction.sd
        allow[test_idx] = prediction.allow
        versions = prediction.versions

    oof = Prediction(s_hat, c_hat, sd, allow, versions)
    picks = pick_all_tiers(
        config, oof, dataset.keys, budget_multipliers(dataset.policy)
    )
    return evaluate(dataset, picks)
