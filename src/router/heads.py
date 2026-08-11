# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""[Q] 점수 헤드, [C] 비용 헤드, [G] 게이트의 구현과 레지스트리.

계약만 고정하고 구현은 이름으로 꽂는다 (DESIGN.md §5-B). 새 트랙을 붙일 때
파이프라인이나 러너를 고칠 필요가 없어야 한다. 고쳐야 한다면 계약이 잘못
그어진 것이다.

계약
    ScoreHead.fit(train) -> None ; predict(texts) -> s_hat[n, 3]
    CostHead.fit(train)  -> None ; predict(texts) -> (c_hat[n, 3], sd[n, 3])
    Gate.fit(train)      -> None ; allow(texts, s_hat, c_hat) -> bool[n, 3]

세 계약 모두 **프롬프트 텍스트만** 받는다. 문항 ID·순서·등급은 인자에 없다.
"""

from __future__ import annotations

from typing import Callable, Dict, Protocol, Sequence

import numpy as np

from .data import MODEL_IDS, Dataset, cost_from_tokens
from .features import FAMILIES, family_codes

N_MODELS = len(MODEL_IDS)


class ScoreHead(Protocol):
    version: str

    def fit(self, train: Dataset) -> None: ...

    def predict(self, texts: Sequence[str]) -> np.ndarray: ...


class CostHead(Protocol):
    version: str

    def fit(self, train: Dataset) -> None: ...

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


class Gate(Protocol):
    version: str

    def fit(self, train: Dataset) -> None: ...

    def allow(
        self, texts: Sequence[str], s_hat: np.ndarray, c_hat: np.ndarray
    ) -> np.ndarray: ...


SCORE_HEADS: Dict[str, Callable[..., ScoreHead]] = {}
COST_HEADS: Dict[str, Callable[..., CostHead]] = {}
GATES: Dict[str, Callable[..., Gate]] = {}


def register(table: Dict[str, Callable], name: str):
    def decorate(factory):
        if name in table:
            raise ValueError(f"이미 등록된 이름: {name}")
        table[name] = factory
        return factory

    return decorate


# --------------------------------------------------------------------------
# 점수 헤드
# --------------------------------------------------------------------------


@register(SCORE_HEADS, "global")
class GlobalScore:
    """전체 평균. 아무 신호도 안 쓰는 하한선."""

    version = "global.v1"

    def fit(self, train: Dataset) -> None:
        self._mean = train.score.mean(axis=0)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.tile(self._mean, (len(texts), 1))


@register(SCORE_HEADS, "family")
class FamilyScore:
    """계열 평균. 계열이 ROI를 지배하므로 이것만으로도 상당히 간다."""

    version = "family.v1"

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        self._table = np.zeros((len(FAMILIES), N_MODELS))
        overall = train.score.mean(axis=0)
        for f in range(len(FAMILIES)):
            m = codes == f
            self._table[f] = train.score[m].mean(axis=0) if m.any() else overall

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self._table[family_codes(texts)]


# --------------------------------------------------------------------------
# 비용 헤드
# --------------------------------------------------------------------------


@register(COST_HEADS, "family")
class FamilyCost:
    """계열별 출력 토큰 분위 + 문자 수 선형회귀로 만든 입력 토큰.

    ``quantile``이 클수록 비용을 위로 틀리게 예측한다. 예산 초과가 0점이므로
    편향 방향은 항상 위쪽이어야 한다 (RULES C2).
    """

    def __init__(self, quantile: float = 0.5) -> None:
        self.quantile = float(quantile)
        self.version = f"familycost.v1(q={self.quantile:g})"

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        chars = np.array([len(t) for t in train.texts], dtype=float)

        # 입력 토큰은 문자 수와 거의 선형이고 모델 간 상관이 0.9998이다.
        design = np.stack([np.ones_like(chars), chars], axis=1)
        self._in_coef = np.linalg.lstsq(design, train.input_tokens, rcond=None)[0]

        self._out = np.zeros((len(FAMILIES), N_MODELS))
        self._out_sd = np.zeros((len(FAMILIES), N_MODELS))
        overall = np.quantile(train.output_tokens, self.quantile, axis=0)
        overall_sd = train.output_tokens.std(axis=0)
        for f in range(len(FAMILIES)):
            m = codes == f
            if m.any():
                self._out[f] = np.quantile(train.output_tokens[m], self.quantile, axis=0)
                self._out_sd[f] = train.output_tokens[m].std(axis=0)
            else:
                self._out[f] = overall
                self._out_sd[f] = overall_sd
        self._policy = train.policy

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        chars = np.array([len(t) for t in texts], dtype=float)
        design = np.stack([np.ones_like(chars), chars], axis=1)
        tok_in = np.maximum(design @ self._in_coef, 1.0)
        codes = family_codes(texts)
        tok_out = self._out[codes]
        cost = cost_from_tokens(tok_in, tok_out, self._policy)
        # 비용의 산포는 출력 토큰 산포에서 온다. 입력 토큰은 잘 맞는다.
        sd = cost_from_tokens(np.zeros_like(tok_in), self._out_sd[codes], self._policy)
        return cost, sd


# --------------------------------------------------------------------------
# 게이트
# --------------------------------------------------------------------------


@register(GATES, "none")
class NoGate:
    version = "none.v1"

    def fit(self, train: Dataset) -> None:
        return None

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        return np.ones((len(texts), N_MODELS), dtype=bool)


@register(GATES, "family_roi")
class FamilyRoiGate:
    """K1 ROI가 낮은 계열의 K1을 차단한다.

    Train 기준 ROI는 `mcq_ko` 6.55 ~ `logic` 0.08로 80배 차이가 난다.
    낮은 쪽을 막는 것만으로 예산이 풀린다.
    """

    def __init__(self, min_roi: float = 1.0) -> None:
        self.min_roi = float(min_roi)
        self.version = f"familyroi.v1(min={self.min_roi:g})"

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        self._roi = np.zeros((len(FAMILIES), N_MODELS))
        for f in range(len(FAMILIES)):
            m = codes == f
            if not m.any():
                continue
            base_s = train.score[m, 0].mean()
            base_c = train.cost[m, 0].mean()
            for j in range(1, N_MODELS):
                extra = train.cost[m, j].mean() - base_c
                self._roi[f, j] = (
                    (train.score[m, j].mean() - base_s) / extra if extra > 0 else 0.0
                )

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        codes = family_codes(texts)
        allow = self._roi[codes] >= self.min_roi
        allow[:, 0] = True
        return allow


@register(GATES, "k1_cost_cap")
class K1CostCapGate:
    """예측 출력 토큰이 상위 분위를 넘는 문항의 K1을 차단한다.

    K1 비용은 light 대비 5.2배~566배로 꼬리가 두껍다. 마진으로는 못 막고
    꼬리를 직접 잘라야 한다 (RULES C3).
    """

    def __init__(self, percentile: float = 90.0, min_roi: float = 1.0) -> None:
        self.percentile = float(percentile)
        self.min_roi = float(min_roi)
        self.version = f"k1cap.v1(p={self.percentile:g},roi={self.min_roi:g})"
        self._roi_gate = FamilyRoiGate(min_roi=min_roi)

    def fit(self, train: Dataset) -> None:
        self._roi_gate.fit(train)
        ratio = train.cost[:, 2] / np.maximum(train.cost[:, 0], 1e-12)
        self._cap = float(np.percentile(ratio, self.percentile))

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        allow = self._roi_gate.allow(texts, s_hat, c_hat)
        ratio = c_hat[:, 2] / np.maximum(c_hat[:, 0], 1e-12)
        allow[:, 2] &= ratio <= self._cap
        allow[:, 0] = True
        return allow


def build_score_head(spec) -> ScoreHead:
    return _build(SCORE_HEADS, spec, "score")


def build_cost_head(spec) -> CostHead:
    return _build(COST_HEADS, spec, "cost")


def build_gate(spec) -> Gate:
    return _build(GATES, spec, "gate")


def _build(table: Dict[str, Callable], spec, label: str):
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        spec = dict(spec)
        name = spec.pop("name")
        kwargs = spec
    if name not in table:
        raise KeyError(f"알 수 없는 {label} 구현: {name!r} (등록됨: {sorted(table)})")
    return table[name](**kwargs)
