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
from .features import FAMILIES, extract, family_codes

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

    def state(self) -> dict:
        return {"table": self._table.tolist()}

    def load_state(self, state: dict) -> None:
        self._table = np.asarray(state["table"], dtype=float)


def _useless_target(train: Dataset) -> np.ndarray:
    """어떤 모델도 light를 넘지 못하는 문항 = 1.

    Train 기준 64.5%가 여기 해당한다(전부 만점 46.7% + 전부 0점 9.0% + 나머지).
    이 문항들에 쓰는 돈은 전부 순손실이다.
    """

    return ((train.score.max(axis=1) - train.score[:, 0]) <= 0).astype(float)


class _UselessModel:
    """계열별 회귀로 '승격이 무의미할 확률'을 예측한다."""

    def fit(self, train: Dataset, alpha: float, min_family: int) -> None:
        codes = family_codes(train.texts)
        features = extract(train.texts)
        target = _useless_target(train)
        self.overall = float(target.mean())
        self.global_fit = _ridge(features, target, alpha)
        self.by_family: Dict[int, tuple] = {}
        self.mean_by_family: Dict[int, float] = {}
        for f in range(len(FAMILIES)):
            m = codes == f
            if not m.any():
                continue
            self.mean_by_family[f] = float(target[m].mean())
            if m.sum() >= min_family:
                self.by_family[f] = _ridge(features[m], target[m], alpha)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        features = extract(texts)
        codes = family_codes(texts)
        out = np.full(len(texts), self.overall)
        for f in np.unique(codes):
            m = codes == f
            key = int(f)
            if key in self.by_family:
                out[m] = _ridge_apply(features[m], self.by_family[key])
            elif key in self.mean_by_family:
                out[m] = self.mean_by_family[key]
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        def pack(fit):
            m, s_, w = fit
            return [m.tolist(), s_.tolist(), w.tolist()]

        return {
            "overall": self.overall,
            "global": pack(self.global_fit),
            "by_family": {str(k): pack(v) for k, v in self.by_family.items()},
            "mean_by_family": {str(k): v for k, v in self.mean_by_family.items()},
        }

    def load_state(self, state: dict) -> None:
        def unpack(row):
            return tuple(np.asarray(x, dtype=float) for x in row)

        self.overall = float(state["overall"])
        self.global_fit = unpack(state["global"])
        self.by_family = {int(k): unpack(v) for k, v in state["by_family"].items()}
        self.mean_by_family = {int(k): float(v) for k, v in state["mean_by_family"].items()}


@register(SCORE_HEADS, "family_useful")
class FamilyUsefulScore:
    """계열 평균 이득을 **문항별 '유용할 확률'로 축소**한다.

    계열 평균 점수 헤드는 같은 계열의 모든 문항에 같은 이득을 준다. 그래서
    배분기가 계열 안에서 우선순위를 매길 수 없다. 문항의 64.5%는 어떤 모델도
    light를 못 넘기는데, 그 구분을 전혀 못 하고 있었다.

    ``ŝ_m = ŝ_light + P(유용) × (계열평균 이득)``

    하드 차단보다 이쪽이 낫다. 배분기의 ROI 정렬이 그대로 살아 있어서
    "유용할 확률이 낮지만 아주 싼" 문항은 여전히 뽑힐 수 있다.
    """

    def __init__(self, alpha: float = 5.0, min_family: int = 40,
                 strength: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self.strength = float(strength)
        self.version = f"familyuseful.v1(a={self.alpha:g},s={self.strength:g})"

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        self._table = np.zeros((len(FAMILIES), N_MODELS))
        overall = train.score.mean(axis=0)
        for f in range(len(FAMILIES)):
            m = codes == f
            self._table[f] = train.score[m].mean(axis=0) if m.any() else overall
        self._useless = _UselessModel()
        self._useless.fit(train, self.alpha, self.min_family)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        base = self._table[family_codes(texts)]
        useful = 1.0 - self.strength * self._useless.predict(texts)
        gain = base[:, 1:] - base[:, [0]]
        out = base.copy()
        out[:, 1:] = base[:, [0]] + gain * useful[:, None]
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {"table": self._table.tolist(), "useless": self._useless.state()}

    def load_state(self, state: dict) -> None:
        self._table = np.asarray(state["table"], dtype=float)
        self._useless = _UselessModel()
        self._useless.load_state(state["useless"])


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

    def state(self) -> dict:
        return {
            "in_coef": self._in_coef.tolist(),
            "out": self._out.tolist(),
            "out_sd": self._out_sd.tolist(),
        }

    def load_state(self, state: dict, policy) -> None:
        self._in_coef = np.asarray(state["in_coef"], dtype=float)
        self._out = np.asarray(state["out"], dtype=float)
        self._out_sd = np.asarray(state["out_sd"], dtype=float)
        self._policy = policy


def _ridge(design: np.ndarray, target: np.ndarray, alpha: float) -> tuple:
    """표준화 → 절편 추가 → 절편만 규제 제외.

    순서를 바꿔 절편까지 표준화하면 절편이 0이 되어 예측이 평균을 못 맞춘다.
    로그 토큰처럼 평균이 6~8인 목표에서는 그 실수 하나로 R²가 -40까지 떨어진다.
    """

    mu = design.mean(axis=0)
    sd = design.std(axis=0) + 1e-9
    scaled = np.hstack([(design - mu) / sd, np.ones((len(design), 1))])
    penalty = np.eye(scaled.shape[1])
    penalty[-1, -1] = 0.0
    weight = np.linalg.solve(scaled.T @ scaled + alpha * penalty, scaled.T @ target)
    return mu, sd, weight


def _ridge_apply(design: np.ndarray, fitted: tuple) -> np.ndarray:
    mu, sd, weight = fitted
    return np.hstack([(design - mu) / sd, np.ones((len(design), 1))]) @ weight


@register(COST_HEADS, "ridge")
class RidgeCost:
    """계열별 회귀로 문항 단위 출력 토큰을 예측한다.

    계열 평균은 계열 안에서 상수라 "비싼 문항만 골라 빼기"가 작동하지 않는다.
    실제 K1 비용 배율은 같은 계열 안에서도 600배씩 벌어진다.

    편향은 **light 대비 상대적으로** 걸어야 한다. 예산 한도의 분모가 light의
    예측 비용이므로, 모든 모델을 똑같이 부풀리면 쓸 수 있는 돈까지 같이 늘어나
    오히려 위험해진다(실측: z를 0 → 1.28로 올렸더니 세 등급이 전부 터졌다).

    - ``z``       승격 모델(ax31·K1)에 얹는 상방 편향. 클수록 승격이 비싸 보인다.
    - ``z_light`` light에 얹는 편향. **음수가 안전하다** — 분모를 낮게 잡으면
      한도가 줄고 승격의 상대 가격도 올라가 양쪽으로 보수적이 된다.

    0.67이면 대략 75분위, 1.28이면 90분위에 해당한다 (RULES C2).

    ``smearing``은 로그 공간 회귀가 **합계를 과소추정**하는 것을 바로잡는다.
    예산은 총액으로 걸리는데 비용은 꼬리가 두꺼워 총액을 소수의 큰 문항이
    지배한다. 로그 평균을 지수로 되돌리면 그 합이 체계적으로 낮게 나온다
    (Jensen). 실측: 개별 문항은 중앙 2.18배로 과대추정하는데 선택된 문항의
    합계는 1.24배로 과소추정했다. Duan의 smearing 추정량
    ``exp(μ̂) · mean(exp(잔차))``로 보정한다.
    """

    def __init__(
        self,
        z: float = 0.0,
        z_light: float = 0.0,
        alpha: float = 3.0,
        min_family: int = 40,
        smearing: bool = True,
    ) -> None:
        self.z = float(z)
        self.z_light = float(z_light)
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self.smearing = bool(smearing)
        self.version = (
            f"ridgecost.v3(z={self.z:g},zl={self.z_light:g},"
            f"a={self.alpha:g},sm={int(self.smearing)})"
        )

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        features = extract(train.texts)
        chars = np.array([len(t) for t in train.texts], dtype=float)

        design_in = np.stack([np.ones_like(chars), chars], axis=1)
        self._in_coef = np.linalg.lstsq(design_in, train.input_tokens, rcond=None)[0]

        log_out = np.log1p(train.output_tokens)
        self._global = [_ridge(features, log_out[:, j], self.alpha) for j in range(N_MODELS)]
        self._global_sd = log_out.std(axis=0)
        global_residual = np.stack(
            [log_out[:, j] - _ridge_apply(features, self._global[j]) for j in range(N_MODELS)],
            axis=1,
        )
        self._global_smear = np.exp(global_residual).mean(axis=0)

        # 외삽 방지. 선형 모델은 학습 범위 밖 특징에서 로그 예측을 20~30까지
        # 밀어올릴 수 있고, expm1을 지나면 토큰 수가 조 단위가 된다. 실제로
        # 한 fold에서 예측 light 비용이 실제의 463배가 나와 세 등급이 전부
        # 터졌다. 비공개 평가에서 이상한 프롬프트 하나면 같은 일이 벌어진다.
        self._log_lo = log_out.min(axis=0)
        self._log_hi = log_out.max(axis=0)
        self._token_hi = train.output_tokens.max(axis=0) * 2.0
        self._in_lo = float(train.input_tokens.min())
        self._in_hi = float(train.input_tokens.max()) * 2.0

        self._by_family: Dict[int, list] = {}
        self._sd_by_family: Dict[int, np.ndarray] = {}
        self._smear_by_family: Dict[int, np.ndarray] = {}
        for f in range(len(FAMILIES)):
            m = codes == f
            if m.sum() < self.min_family:
                continue
            fitted = [_ridge(features[m], log_out[m, j], self.alpha) for j in range(N_MODELS)]
            residual = np.stack(
                [log_out[m, j] - _ridge_apply(features[m], fitted[j]) for j in range(N_MODELS)],
                axis=1,
            )
            self._by_family[f] = fitted
            self._sd_by_family[f] = residual.std(axis=0)
            self._smear_by_family[f] = np.exp(residual).mean(axis=0)
        self._policy = train.policy

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        features = extract(texts)
        codes = family_codes(texts)
        chars = np.array([len(t) for t in texts], dtype=float)
        tok_in = np.clip(
            np.stack([np.ones_like(chars), chars], axis=1) @ self._in_coef,
            self._in_lo,
            self._in_hi,
        )

        log_out = np.zeros((len(texts), N_MODELS))
        sd_log = np.zeros((len(texts), N_MODELS))
        smear = np.ones((len(texts), N_MODELS))
        for f in np.unique(codes):
            m = codes == f
            fitted = self._by_family.get(int(f), self._global)
            spread = self._sd_by_family.get(int(f), self._global_sd)
            for j in range(N_MODELS):
                log_out[m, j] = _ridge_apply(features[m], fitted[j])
            sd_log[m] = spread
            if self.smearing:
                smear[m] = self._smear_by_family.get(int(f), self._global_smear)

        # 회귀 출력을 학습에서 본 범위로 먼저 묶은 뒤 편향을 얹는다.
        log_out = np.clip(log_out, self._log_lo, self._log_hi)
        bias = np.full(N_MODELS, self.z)
        bias[0] = self.z_light
        # smearing은 exp(잔차)의 평균이므로 log1p 되돌리기 전에 곱한다.
        tok_out = np.clip(
            np.exp(log_out + bias * sd_log) * smear - 1.0, 0.0, self._token_hi
        )
        cost = cost_from_tokens(tok_in, tok_out, self._policy)
        # 산포는 로그 공간 잔차를 비용 단위로 옮겨 근사한다.
        spread_tokens = np.clip(
            np.expm1(log_out + sd_log) - np.expm1(log_out), 0.0, self._token_hi
        )
        sd = cost_from_tokens(np.zeros_like(tok_in), spread_tokens, self._policy)
        return cost, sd

    def state(self) -> dict:
        def pack(fitted):
            return [[m.tolist(), s.tolist(), w.tolist()] for (m, s, w) in fitted]

        return {
            "in_coef": self._in_coef.tolist(),
            "global": pack(self._global),
            "global_sd": self._global_sd.tolist(),
            "global_smear": self._global_smear.tolist(),
            "by_family": {
                str(f): pack(v) for f, v in self._by_family.items()
            },
            "sd_by_family": {
                str(f): v.tolist() for f, v in self._sd_by_family.items()
            },
            "smear_by_family": {
                str(f): v.tolist() for f, v in self._smear_by_family.items()
            },
            "log_lo": self._log_lo.tolist(),
            "log_hi": self._log_hi.tolist(),
            "token_hi": self._token_hi.tolist(),
            "in_lo": self._in_lo,
            "in_hi": self._in_hi,
        }

    def load_state(self, state: dict, policy) -> None:
        def unpack(rows):
            return [
                (np.asarray(m, dtype=float), np.asarray(s, dtype=float),
                 np.asarray(w, dtype=float))
                for (m, s, w) in rows
            ]

        self._in_coef = np.asarray(state["in_coef"], dtype=float)
        self._global = unpack(state["global"])
        self._global_sd = np.asarray(state["global_sd"], dtype=float)
        self._global_smear = np.asarray(state["global_smear"], dtype=float)
        self._by_family = {int(k): unpack(v) for k, v in state["by_family"].items()}
        self._sd_by_family = {
            int(k): np.asarray(v, dtype=float) for k, v in state["sd_by_family"].items()
        }
        self._smear_by_family = {
            int(k): np.asarray(v, dtype=float)
            for k, v in state["smear_by_family"].items()
        }
        self._log_lo = np.asarray(state["log_lo"], dtype=float)
        self._log_hi = np.asarray(state["log_hi"], dtype=float)
        self._token_hi = np.asarray(state["token_hi"], dtype=float)
        self._in_lo = float(state["in_lo"])
        self._in_hi = float(state["in_hi"])
        self._policy = policy


# --------------------------------------------------------------------------
# 게이트
# --------------------------------------------------------------------------


@register(GATES, "none")
class NoGate:
    version = "none.v1"

    def fit(self, train: Dataset) -> None:
        return None

    def state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
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

    def state(self) -> dict:
        return {"roi": self._roi.tolist()}

    def load_state(self, state: dict) -> None:
        self._roi = np.asarray(state["roi"], dtype=float)


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

    def state(self) -> dict:
        return {"cap": self._cap, "roi": self._roi_gate.state()}

    def load_state(self, state: dict) -> None:
        self._cap = float(state["cap"])
        self._roi_gate.load_state(state["roi"])


@register(GATES, "runaway_guard")
class RunawayGuardGate:
    """생성이 폭주할 것 같은 문항의 승격을 막는다.

    파산의 대부분은 예측 실패 몇 건이 만든다. 실측: Fast 파산 트라이얼에서
    추가비용의 83~89%를 단일 문항 하나가 만들었고, 그 문항은 실제 비용이
    예측의 36배였다. 모델이 답을 못 내고 수만 토큰을 뱉은 경우다.

    이런 폭주는 로그 회귀의 조건부 평균으로는 절대 못 맞힌다. 대신 **폭주는
    모델을 가리지 않는다** — 계열 안에서 light 출력과 ax31 출력의 상관이
    0.75~0.78이다. 그리고 light 출력은 비교적 잘 예측된다(R² 0.61).

    그래서 "light가 이미 많이 뱉을 것 같은 문항"을 승격 후보에서 뺀다.
    개별 문항의 비용을 위로 크게 부풀리는 방식은 전체를 망가뜨리는데,
    이 방식은 꼬리만 잘라낸다.
    """

    def __init__(self, percentile: float = 97.0, inner: str = "k1_cost_cap",
                 **inner_kwargs) -> None:
        self.percentile = float(percentile)
        self._inner = GATES[inner](**inner_kwargs)
        self.version = f"runaway.v1(p={self.percentile:g})+{self._inner.version}"

    def fit(self, train: Dataset) -> None:
        self._inner.fit(train)
        # 임계값은 **예측 분포**에서 잡아야 한다. 실제 비용 분위로 잡으면
        # 예측이 평균으로 수축돼 있어 아무 문항도 안 걸린다(실측: 효과 0).
        # 예측 상위 5%가 실제 폭주의 85%를 잡는다.
        from .features import extract  # noqa: F401  (헤드 내부 일관성용)

        self._threshold = 0.0
        self._threshold_percentile = self.percentile

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        allow = self._inner.allow(texts, s_hat, c_hat)
        # 배치 안의 예측 분포에서 자른다. 배치 구성은 규칙상 볼 수 있는
        # 정보이고, 문항 ID나 입력 순서를 쓰는 것과 다르다.
        cut = float(np.percentile(c_hat[:, 0], self._threshold_percentile))
        allow[c_hat[:, 0] > cut, 1:] = False
        allow[:, 0] = True
        return allow

    def state(self) -> dict:
        return {"percentile": self._threshold_percentile, "inner": self._inner.state()}

    def load_state(self, state: dict) -> None:
        self._threshold_percentile = float(state["percentile"])
        self._inner.load_state(state["inner"])


@register(GATES, "useless_block")
class UselessBlockGate:
    """승격이 무의미할 확률 상위 ``percentile``%의 승격을 하드 차단한다.

    ``family_useful`` 점수 헤드의 하드 버전이다. 어느 쪽이 나은지는 실험으로
    정한다 — 하드 차단은 싼 문항까지 같이 버리고, 축소는 그걸 살린다.
    """

    def __init__(self, percentile: float = 80.0, alpha: float = 5.0,
                 min_family: int = 40, inner: str = "k1_cost_cap",
                 **inner_kwargs) -> None:
        self.percentile = float(percentile)
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self._inner = GATES[inner](**inner_kwargs)
        self.version = f"uselessblock.v1(p={self.percentile:g})+{self._inner.version}"

    def fit(self, train: Dataset) -> None:
        self._inner.fit(train)
        self._useless = _UselessModel()
        self._useless.fit(train, self.alpha, self.min_family)

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        allow = self._inner.allow(texts, s_hat, c_hat)
        risk = self._useless.predict(texts)
        cut = float(np.percentile(risk, self.percentile))
        allow[risk >= cut, 1:] = False
        allow[:, 0] = True
        return allow

    def state(self) -> dict:
        return {"percentile": self.percentile, "useless": self._useless.state(),
                "inner": self._inner.state()}

    def load_state(self, state: dict) -> None:
        self.percentile = float(state["percentile"])
        self._useless = _UselessModel()
        self._useless.load_state(state["useless"])
        self._inner.load_state(state["inner"])


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
