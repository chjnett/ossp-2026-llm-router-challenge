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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .allocate import allocate
from .data import TIERS, Dataset, budget_multipliers
from .harness import Evaluation, evaluate
from .heads import build_cost_head, build_gate, build_score_head

DEFAULT_UTIL = {"fast": 0.90, "balanced": 0.90, "premium": 0.85}


@dataclass(frozen=True)
class Config:
    id: str
    score: Any = "family"
    cost: Any = "family"
    gate: Any = "none"
    alloc: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @staticmethod
    def load(path: Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(raw) - {"id", "score", "cost", "gate", "alloc", "note"}
        if unknown:
            raise ValueError(f"알 수 없는 설정 키: {sorted(unknown)}")
        return Config(**raw)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "cost": self.cost,
            "gate": self.gate,
            "alloc": self.alloc,
            "note": self.note,
        }

    @property
    def util(self) -> Dict[str, float]:
        raw = self.alloc.get("util", DEFAULT_UTIL)
        if isinstance(raw, (int, float)):
            return {t: float(raw) for t in TIERS}
        merged = dict(DEFAULT_UTIL)
        merged.update({k: float(v) for k, v in raw.items()})
        return merged

    @property
    def mu(self) -> float:
        return float(self.alloc.get("mu", 0.0))

    @property
    def headroom(self) -> Dict[str, float] | None:
        """여윳돈을 얼마나 쓸지. **등급 간 비교가 되는 유일한 안전 손잡이다.**

        ``util``은 전체 예산 대비 비율이라 같은 값이 등급마다 전혀 다른 뜻이
        된다. ``util=0.9``면 Fast는 여윳돈 0.25 중 0.125(50%)를 쓰지만
        Premium은 3.0 중 2.6(87%)을 쓴다. 그래서 util에 일률적으로 -0.03을
        빼면 Fast만 여유가 통째로 사라진다(실측: Fast가 all-light로 퇴화).

        ``headroom=h``는 세 등급 모두 여윳돈의 h를 쓴다는 뜻이다.
        실효 배율은 ``1 + h * (배율 - 1)``이 된다.
        """

        raw = self.alloc.get("headroom")
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return {t: float(raw) for t in TIERS}
        merged = {t: 0.85 for t in TIERS}
        merged.update({k: float(v) for k, v in raw.items()})
        return merged

    @property
    def size_penalty(self) -> float:
        """배치가 작을수록 여유를 더 준다.

        실현 비용 비율의 흔들림은 표본 수의 제곱근에 반비례한다. 200문항
        배치에서 파산이 몰리는 것이 그 때문이다. 배치 구성은 규칙상 볼 수 있는
        정보이고, 문항 ID나 입력 순서를 쓰는 것과는 다르다.
        """

        return float(self.alloc.get("size_penalty", 0.0))


def effective_util(
    config: Config, n_episodes: int, multipliers: Mapping[str, float]
) -> Dict[str, float]:
    """할당기에 넘길 등급별 사용률을 만든다.

    ``headroom``이 지정되면 그것을 우선 쓰고, 없으면 예전 ``util``을 쓴다.
    반환값은 항상 '전체 예산 대비 비율'이라 할당기 쪽은 바뀌지 않는다.
    """

    penalty = config.size_penalty / max(1.0, float(n_episodes)) ** 0.5
    headroom = config.headroom
    if headroom is None:
        return {t: max(0.0, u - penalty) for t, u in config.util.items()}
    result = {}
    for tier, h in headroom.items():
        multiplier = float(multipliers[tier])
        adjusted = max(0.0, h - penalty)
        result[tier] = (1.0 + adjusted * (multiplier - 1.0)) / multiplier
    return result


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

    Dev 점수는 보정과 통과 여부 확인용이고, 어느 구성이 나은지는 Train CV로
    정한다. Dev에 맞춰 고르기 시작하면 Dev가 학습 자료가 된다.
    """

    folds = dataset.folds(k)
    multipliers = budget_multipliers(dataset.policy)
    oof = {tier: np.zeros(len(dataset), dtype=int) for tier in TIERS}

    for f, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        fit_part = dataset.subset(train_idx)
        target = dataset.subset(test_idx)
        prediction = predict(config, fit_part, target.texts)
        picks = pick_all_tiers(config, prediction, target.keys, multipliers)
        for tier in TIERS:
            oof[tier][test_idx] = picks[tier]

    return evaluate(dataset, oof)
