# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""[A] λ 할당기와 [S] 안전 검증.

목적함수:  선택 = argmax_m [ ŝ − λ·ĉ − μ·sd(ĉ) ]

λ는 고정값이 아니라 **입력 배치 안에서 이분탐색**해 등급 예산에 맞춘다.
고정 λ는 분포가 바뀌면 무너진다 — 공식 hash-regex baseline이 죽은 지점이다.

순서 불변성(RULES B)이 이 모듈의 정확성 요건이다.
평가 재실행에서 문항 ID와 순서가 전부 바뀐 입력이 한 번 더 들어오고,
선택이 하나라도 다르면 자동 채점이 멈춘다.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .envelope import (
    Segments,
    build_segments,
    take_chance_constrained,
    take_monte_carlo,
    take_within_budget,
)

NEG = -np.inf


def order_invariant_sum(values: np.ndarray) -> float:
    """순서와 무관하게 같은 값을 주는 합.

    ``math.fsum``은 정확한 합을 구한 뒤 한 번만 반올림하므로 어떤 순열에도
    같은 결과를 준다. ``np.sum``의 pairwise 누적은 그 보장이 없다.
    """

    return math.fsum(np.asarray(values, dtype=float).ravel().tolist())


@dataclass(frozen=True)
class AllocationPlan:
    picks: np.ndarray            # [n] MODEL_IDS 인덱스
    lam: float                   # 확정된 λ
    estimated_ratio: float       # 예측 비용 기준 비율
    demoted: int                 # [S]에서 강등된 문항 수
    hit_floor: bool              # λ 하한에서도 예산이 남았는가 (예산 여유)


def utility(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    lam: float,
    *,
    sd: np.ndarray | None = None,
    mu: float = 0.0,
) -> np.ndarray:
    u = s_hat - lam * c_hat
    if sd is not None and mu:
        u = u - mu * sd
    return u


def _pick(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    lam: float,
    allow: np.ndarray,
    sd: np.ndarray | None,
    mu: float,
) -> np.ndarray:
    u = utility(s_hat, c_hat, lam, sd=sd, mu=mu)
    u = np.where(allow, u, NEG)
    # MODEL_IDS가 싼 것부터 정렬돼 있으므로 argmax의 first-max 규칙이
    # 곧 "동률이면 싼 쪽"이 된다 (RULES B3). 순서·ID와 무관하다.
    return np.argmax(u, axis=1)


def _row_cost(c_hat: np.ndarray, picks: np.ndarray) -> np.ndarray:
    return c_hat[np.arange(len(picks)), picks]


def allocate(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    *,
    multiplier: float,
    util: float = 0.90,
    allow: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    mu: float = 0.0,
    keys: Sequence[str] | None = None,
) -> AllocationPlan:
    """예산 안에서 문항별 모델을 고른다 (오목 포락선 그리디).

    λ 이분탐색과 같은 답을 주지만 정렬 한 번으로 끝나 파산 게이트를 수천 번
    돌릴 수 있다. 두 구현이 일치하는지는 테스트로 고정한다.

    ``multiplier``는 정책의 예산 배율, ``util``은 그중 실제로 쓸 목표 비율이다.
    분모(light 기준 비용)도 예측값이라는 점에 주의한다 — 평가 시점에는
    실제 light 비용도 알 수 없다.
    """

    n = len(s_hat)
    effective = s_hat if (sd is None or not mu) else s_hat - mu * sd
    if keys is None:
        # 위치 기반 기본키를 쓰면 입력 순서가 동률 처리에 새어 들어간다.
        # 예측값 자체에서 키를 만들면 예측이 같은 문항은 같은 그룹이 되어
        # 통째로 승격되거나 통째로 남는다.
        keys = [
            hashlib.blake2b(
                np.ascontiguousarray(
                    np.concatenate([effective[i], c_hat[i]])
                ).tobytes(),
                digest_size=8,
            ).hexdigest()
            for i in range(n)
        ]

    # 예산 한도는 정책상 **light 전 문항 비용**이 기준이다. 출발 배분은 그보다
    # 쌀 수 있으므로(가장 싼 모델에서 시작) 여유를 그 차이만큼 더 잡는다.
    base = order_invariant_sum(c_hat[:, 0])
    segments = build_segments(effective, c_hat, list(keys), allow=allow)
    start_cost = order_invariant_sum(
        c_hat[np.arange(n), segments.base_model]
    )
    budget_extra = base * multiplier * util - start_cost
    picks = take_within_budget(segments, n, budget_extra)
    used = order_invariant_sum(_row_cost(c_hat, picks))
    return AllocationPlan(
        picks=picks,
        lam=_implied_lambda(segments, picks),
        estimated_ratio=used / base,
        demoted=0,
        hit_floor=bool(len(segments.order)) and used < base * multiplier * util,
    )


def allocate_chance(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    cost_sd: np.ndarray,
    *,
    multiplier: float,
    epsilon: float = 0.01,
    allow: np.ndarray | None = None,
    keys: Sequence[str] | None = None,
) -> AllocationPlan:
    """총합 초과확률을 ε 이하로 묶는 배분.

    ``c_hat``은 **불편추정**이어야 한다. 여기에 상방 편향을 또 얹으면
    안전이 두 번 걸려 예산을 못 쓰게 된다.
    """

    n = len(s_hat)
    if keys is None:
        keys = [
            hashlib.blake2b(
                np.ascontiguousarray(np.concatenate([s_hat[i], c_hat[i]])).tobytes(),
                digest_size=8,
            ).hexdigest()
            for i in range(n)
        ]

    variance = np.asarray(cost_sd, dtype=float) ** 2
    segments = build_segments(
        s_hat, c_hat, list(keys), allow=allow, variance=variance
    )
    mean_light = order_invariant_sum(c_hat[:, 0])
    var_light = order_invariant_sum(variance[:, 0])
    picks = take_chance_constrained(
        segments,
        n,
        multiplier=multiplier,
        mean_light=mean_light,
        var_light=var_light,
        z_epsilon=_z_from_epsilon(epsilon),
    )
    used = order_invariant_sum(_row_cost(c_hat, picks))
    return AllocationPlan(
        picks=picks,
        lam=_implied_lambda(segments, picks),
        estimated_ratio=used / mean_light,
        demoted=0,
        hit_floor=False,
    )


def allocate_monte_carlo(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    multipliers_draw: np.ndarray,
    *,
    multiplier: float,
    epsilon: float = 0.01,
    allow: np.ndarray | None = None,
    keys: Sequence[str] | None = None,
) -> AllocationPlan:
    """경험 잔차 재표집으로 초과확률을 직접 제어한다.

    ``multipliers_draw``는 [n, 3, D] 곱셈 잔차다. 난수가 아니라 콘텐츠 해시로
    결정되므로 같은 프롬프트는 항상 같은 표본을 받고 입력 순서와 무관하다.
    """

    n = len(s_hat)
    if keys is None:
        keys = [
            hashlib.blake2b(
                np.ascontiguousarray(np.concatenate([s_hat[i], c_hat[i]])).tobytes(),
                digest_size=8,
            ).hexdigest()
            for i in range(n)
        ]

    segments = build_segments(s_hat, c_hat, list(keys), allow=allow)
    draws = multipliers_draw.shape[2]

    # 표본별 실현 비용. 배분기가 쓰는 평균 예측에 잔차 배율을 곱한다.
    realized = c_hat[:, :, None] * multipliers_draw          # [n, 3, D]
    light_totals = realized[:, 0, :].sum(axis=0)             # [D]

    if len(segments.episode):
        ep, to, fr = segments.episode, segments.to_model, segments.from_model
        seg_draws = realized[ep, to, :] - realized[ep, fr, :]
        seg_draws = np.maximum(seg_draws, 0.0)
    else:
        seg_draws = np.zeros((0, draws))

    picks = take_monte_carlo(
        segments,
        n,
        multiplier=multiplier,
        light_totals=light_totals,
        seg_draws=seg_draws,
        epsilon=epsilon,
    )
    base = order_invariant_sum(c_hat[:, 0])
    used = order_invariant_sum(_row_cost(c_hat, picks))
    return AllocationPlan(
        picks=picks,
        lam=_implied_lambda(segments, picks),
        estimated_ratio=used / base,
        demoted=0,
        hit_floor=False,
    )


def _z_from_epsilon(epsilon: float) -> float:
    """표준정규 상위 ε 분위. 표를 쓰지 않고 유리근사로 구한다."""

    epsilon = min(max(float(epsilon), 1e-6), 0.5)
    # Acklam 근사의 꼬리 구간 형태. ε<=0.5 이므로 상위 분위만 다룬다.
    t = math.sqrt(-2.0 * math.log(epsilon))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )


def _implied_lambda(segments: Segments, picks: np.ndarray) -> float:
    """마지막으로 집힌 구간의 ROI. λ 이분탐색이 수렴했을 값과 같은 뜻이다."""

    if len(segments.order) == 0:
        return 0.0
    taken = [
        k
        for k in segments.order
        if picks[segments.episode[k]] >= segments.to_model[k]
        and picks[segments.episode[k]] != 0
    ]
    return float(segments.roi[taken[-1]]) if taken else float("inf")


def allocate_bisect(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    *,
    multiplier: float,
    util: float = 0.90,
    allow: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    mu: float = 0.0,
    iterations: int = 100,
) -> AllocationPlan:
    """λ 이분탐색 구현. 느리지만 독립적이라 ``allocate``의 교차 검증에 쓴다."""

    n, m = s_hat.shape
    if allow is None:
        allow = np.ones((n, m), dtype=bool)
    allow = allow.copy()
    allow[:, 0] = True  # light는 항상 열어 둔다. 폴백이 없으면 안 된다.

    target = order_invariant_sum(c_hat[:, 0]) * multiplier * util

    # λ 상한: 이 값을 넘으면 어떤 문항도 승격되지 않는다.
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = s_hat[:, 1:] - s_hat[:, [0]]
        extra = c_hat[:, 1:] - c_hat[:, [0]]
        ratio = np.where(extra > 0, gain / np.maximum(extra, 1e-12), 0.0)
    hi = float(max(1e-9, np.nanmax(ratio) * 1.05)) if ratio.size else 1.0
    lo = 0.0

    # λ=0(예산 무시)에서도 한도를 안 넘으면 더 조일 이유가 없다.
    picks_free = _pick(s_hat, c_hat, 0.0, allow, sd, mu)
    if order_invariant_sum(_row_cost(c_hat, picks_free)) <= target:
        picks, lam, hit_floor = picks_free, 0.0, True
    else:
        for _ in range(iterations):
            mid = (lo + hi) / 2
            picks = _pick(s_hat, c_hat, mid, allow, sd, mu)
            if order_invariant_sum(_row_cost(c_hat, picks)) <= target:
                hi = mid
            else:
                lo = mid
        lam, hit_floor = hi, False
        picks = _pick(s_hat, c_hat, lam, allow, sd, mu)

    base = order_invariant_sum(c_hat[:, 0])
    return AllocationPlan(
        picks=picks,
        lam=lam,
        estimated_ratio=order_invariant_sum(_row_cost(c_hat, picks)) / base,
        demoted=0,
        hit_floor=hit_floor,
    )


def safety_demote(
    plan: AllocationPlan,
    s_hat: np.ndarray,
    c_pessimistic: np.ndarray,
    keys: Sequence[str],
    *,
    multiplier: float,
    util: float,
) -> AllocationPlan:
    """[S] 더 비관적인 비용으로 다시 재고, 넘치면 ROI 낮은 순으로 강등한다.

    동률은 **콘텐츠 해시**로 깬다. 입력 순서나 문항 ID로 깨면 감사 재실행에서
    선택이 달라진다.
    """

    picks = plan.picks.copy()
    n = len(picks)
    base = order_invariant_sum(c_pessimistic[:, 0])
    target = base * multiplier * util
    used = order_invariant_sum(_row_cost(c_pessimistic, picks))
    if used <= target:
        return AllocationPlan(picks, plan.lam, plan.estimated_ratio, 0, plan.hit_floor)

    # 승격된 문항의 ROI = (이득) / (추가 비용). 낮은 것부터 되돌린다.
    upgraded = [i for i in range(n) if picks[i] != 0]
    def roi(i: int) -> tuple[float, str]:
        gain = float(s_hat[i, picks[i]] - s_hat[i, 0])
        extra = float(c_pessimistic[i, picks[i]] - c_pessimistic[i, 0])
        return (gain / extra if extra > 0 else -math.inf, keys[i])

    order = sorted(upgraded, key=roi)  # (ROI, 콘텐츠 해시) 오름차순
    demoted = 0
    for i in order:
        if used <= target:
            break
        picks[i] = 0
        demoted += 1
        # 누적 뺄셈 대신 매번 다시 합산한다. 부동소수점 누적 오차가
        # 강등 순서에 따라 달라지는 것을 막는다.
        used = order_invariant_sum(_row_cost(c_pessimistic, picks))

    return AllocationPlan(
        picks=picks,
        lam=plan.lam,
        estimated_ratio=order_invariant_sum(_row_cost(c_pessimistic, picks)) / base,
        demoted=demoted,
        hit_floor=plan.hit_floor,
    )
