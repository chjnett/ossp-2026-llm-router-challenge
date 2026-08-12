# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""승격 구간(segment)과 오목 포락선.

λ를 이분탐색하는 대신, 문항마다 (비용, 점수) 점들의 **오목 포락선**을 구해
"light → … → 더 비싼 모델"로 가는 증분 구간들로 쪼갠다. 구간의 ROI는 한 문항
안에서 반드시 감소하므로, 전체 구간을 ROI 내림차순으로 훑으며 예산이 허락하는
동안 집으면 그것이 곧 λ-스윕의 결과다.

이 방식이 이분탐색보다 나은 점:

* **정확하다.** 경계에서 λ를 근사하지 않는다.
* **빠르다.** 정렬 한 번이면 끝난다. 파산 게이트를 N≥2,000회 돌릴 수 있다.
* **순서 불변이 자명하다.** 정렬 키가 (−ROI, 콘텐츠 해시)뿐이라 입력 순서가
  누적 순서에 영향을 주지 않는다 (RULES B2·B3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segments:
    """모든 문항의 승격 구간을 ROI 내림차순으로 펼쳐 둔 것."""

    episode: np.ndarray     # [k] 문항 인덱스
    from_model: np.ndarray  # [k] 이 구간을 집기 전에 있어야 하는 모델 인덱스
    to_model: np.ndarray    # [k] 이 구간을 집었을 때 도달하는 모델 인덱스
    delta_cost: np.ndarray  # [k] 직전 단계 대비 추가 비용 (> 0)
    delta_gain: np.ndarray  # [k] 직전 단계 대비 점수 상승 (> 0)
    roi: np.ndarray         # [k] delta_gain / delta_cost
    order: np.ndarray       # [k] ROI 내림차순 정렬 인덱스 (동률은 콘텐츠 해시)
    base_model: np.ndarray  # [n] 아무것도 승격하지 않았을 때의 출발 모델
    group: np.ndarray       # [k] order 상의 그룹 번호. 같은 번호는 통째로 처리한다
    delta_var: np.ndarray   # [k] 추가 비용의 분산. 확률 제약 배분에만 쓴다


def build_segments(
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    keys: list[str] | tuple[str, ...],
    *,
    allow: np.ndarray | None = None,
    variance: np.ndarray | None = None,
) -> Segments:
    """문항별 오목 포락선을 구간 목록으로 편다.

    ``allow[i, m]``이 False인 모델은 포락선에서 제외한다. light(열 0)는 항상
    허용한다 — 폴백이 없으면 안 된다.
    """

    n, m = s_hat.shape
    if allow is None:
        allow = np.ones((n, m), dtype=bool)
    allow = allow.copy()
    allow[:, 0] = True

    ep, fr, to, dc, dg = [], [], [], [], []
    base_model = np.zeros(n, dtype=int)
    for i in range(n):
        allowed = [j for j in range(m) if allow[i, j]]
        # 출발점은 light가 아니라 **예측 비용이 가장 싼 모델**이다.
        # 드물지만 ax31이 light보다 싼 문항이 있고(Train 20/1760), 그 문항들이
        # 커서 절감액은 light 총비용의 5.8%에 달한다. light에서 출발하면
        # 이 여유를 통째로 놓친다. 동률이면 싼 쪽 = 인덱스가 작은 쪽.
        cur_j = min(allowed, key=lambda j: (float(c_hat[i, j]), j))
        base_model[i] = cur_j
        cur_c = float(c_hat[i, cur_j])
        cur_s = float(s_hat[i, cur_j])
        remaining = [j for j in allowed if float(c_hat[i, j]) > cur_c]
        while remaining:
            best_j, best_roi = -1, 0.0
            for j in remaining:
                d_c = float(c_hat[i, j]) - cur_c
                d_s = float(s_hat[i, j]) - cur_s
                if d_c <= 0.0 or d_s <= 0.0:
                    continue
                r = d_s / d_c
                # 동률이면 싼 모델을 먼저 (MODEL_IDS가 싼 것부터 정렬돼 있다).
                if r > best_roi:
                    best_j, best_roi = j, r
            if best_j < 0:
                break
            ep.append(i)
            fr.append(cur_j)
            to.append(best_j)
            dc.append(float(c_hat[i, best_j]) - cur_c)
            dg.append(float(s_hat[i, best_j]) - cur_s)
            cur_j = best_j
            cur_c = float(c_hat[i, best_j])
            cur_s = float(s_hat[i, best_j])
            # 모델 인덱스가 아니라 비용으로 거른다. 예측 비용이 뒤집히는
            # 문항에서도 포락선이 깨지지 않는다.
            remaining = [j for j in remaining if float(c_hat[i, j]) > cur_c]

    episode = np.asarray(ep, dtype=int)
    from_model = np.asarray(fr, dtype=int)
    to_model = np.asarray(to, dtype=int)
    delta_cost = np.asarray(dc, dtype=float)
    delta_gain = np.asarray(dg, dtype=float)
    roi = np.divide(
        delta_gain, delta_cost, out=np.zeros_like(delta_gain), where=delta_cost > 0
    )

    if variance is None:
        delta_var = np.zeros_like(delta_cost)
    else:
        # 보수적 근사. Var[c_m - c_light] = Var[c_m] + Var[c_light] - 2Cov 인데
        # 두 비용은 강하게 양의 상관이라 Cov를 무시하면 오히려 위험하다.
        # 그래서 더 큰 쪽인 Var[c_m]을 그대로 쓴다.
        delta_var = (
            variance[episode, to_model] if len(episode) else np.zeros(0)
        )

    if len(episode):
        # 동률은 콘텐츠 해시로만 깬다. 입력 순서로 깨면 감사에서 선택이 갈린다.
        tie = np.array([keys[i] for i in episode])
        order = np.lexsort((to_model, tie, -roi))
        group = _group_boundaries(roi[order], tie[order], to_model[order])
    else:
        order = np.asarray([], dtype=int)
        group = np.asarray([], dtype=int)

    return Segments(
        episode,
        from_model,
        to_model,
        delta_cost,
        delta_gain,
        roi,
        order,
        base_model,
        group,
        delta_var,
    )


def _group_boundaries(
    roi_sorted: np.ndarray, tie_sorted: np.ndarray, to_sorted: np.ndarray
) -> np.ndarray:
    """정렬된 구간 목록에서 '내용이 완전히 같은' 이웃끼리 그룹 번호를 매긴다.

    같은 프롬프트가 배치에 여러 번 있으면 예측도 ROI도 같다. 그중 일부만
    승격하면 '같은 프롬프트와 등급의 선택은 같아야 한다'를 어긴다. 그래서
    같은 그룹은 통째로 승격하거나 통째로 남긴다.
    """

    if len(roi_sorted) == 0:
        return np.asarray([], dtype=int)
    same = (
        (roi_sorted[1:] == roi_sorted[:-1])
        & (tie_sorted[1:] == tie_sorted[:-1])
        & (to_sorted[1:] == to_sorted[:-1])
    )
    return np.concatenate([[0], np.cumsum(~same)])


def take_chance_constrained(
    segments: Segments,
    n_episodes: int,
    *,
    multiplier: float,
    mean_light: float,
    var_light: float,
    z_epsilon: float,
) -> np.ndarray:
    """총합의 초과확률이 ε 이하가 되도록 승격을 고른다.

    지금까지는 비용 예측을 통째로 부풀리고(z) 목표 사용률을 낮춰(headroom)
    안전을 샀다. 그러면 손잡이 하나가 안전과 활용을 반대로 잡아당긴다 —
    실측에서 z=1.28이 승격 비용을 ax31 1.85배·K1 2.72배 부풀려, 배분기는
    예산을 다 썼다고 믿는데 Premium은 한도의 40%만 쓰고 있었다.

    여기서는 비용을 **불편추정**으로 두고 제약만 확률로 건다.

        Σμ + z·√(Σσ² + (배율−1)²·σ²_light)  ≤  (배율−1)·μ_light

    좌변은 승격을 추가할수록 단조 증가하므로 정렬된 앞부분을 취하면 된다.
    배치가 작을수록 분산항의 비중이 커져 자동으로 보수적이 되므로, 크기별
    마진(size_penalty)을 따로 둘 필요가 없다.
    """

    picks = segments.base_model.copy()
    if len(segments.order) == 0:
        return picks

    limit = (multiplier - 1.0) * mean_light
    tail = (multiplier - 1.0) ** 2 * var_light

    idx = segments.order
    grp = segments.group
    episode, from_model = segments.episode, segments.from_model
    to_model, delta_cost = segments.to_model, segments.delta_cost
    delta_var = segments.delta_var

    total_mean = 0.0
    total_var = 0.0
    start, n_seg = 0, len(idx)
    while start < n_seg:
        end = start + 1
        while end < n_seg and grp[end] == grp[start]:
            end += 1
        members = [k for k in idx[start:end] if picks[episode[k]] == from_model[k]]
        if members:
            add_mean = math.fsum(float(delta_cost[k]) for k in members)
            add_var = math.fsum(float(delta_var[k]) for k in members)
            new_mean = total_mean + add_mean
            new_var = total_var + add_var
            if new_mean + z_epsilon * math.sqrt(max(new_var + tail, 0.0)) <= limit:
                for k in members:
                    picks[episode[k]] = to_model[k]
                total_mean, total_var = new_mean, new_var
        start = end
    return picks


def take_monte_carlo(
    segments: Segments,
    n_episodes: int,
    *,
    multiplier: float,
    light_totals: np.ndarray,
    seg_draws: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """초과확률을 **경험 분포 재표집**으로 직접 추정해 승격을 고른다.

    CLT는 이 분포에서 통하지 않는다. 비용 배율이 5~566배로 벌어져 정규
    근사가 꼬리를 통째로 놓친다. 실측: ε=0.001로 잡아도 실제 초과율이 200%가
    넘었다. 그래서 정규 가정을 버리고 학습 잔차를 그대로 다시 뽑는다.

    ``light_totals[d]``는 표본 d에서의 light 총비용, ``seg_draws[k, d]``는
    구간 k의 추가비용 실현값이다. 승격을 하나 더 집었을 때 D개 표본 중
    한도를 넘는 비율이 ``epsilon``을 넘으면 멈춘다.
    """

    picks = segments.base_model.copy()
    if len(segments.order) == 0:
        return picks

    limits = (multiplier - 1.0) * light_totals
    running = np.zeros_like(limits)

    idx = segments.order
    grp = segments.group
    episode, from_model = segments.episode, segments.from_model
    to_model = segments.to_model

    start, n_seg = 0, len(idx)
    while start < n_seg:
        end = start + 1
        while end < n_seg and grp[end] == grp[start]:
            end += 1
        members = [k for k in idx[start:end] if picks[episode[k]] == from_model[k]]
        if members:
            candidate = running + seg_draws[members].sum(axis=0)
            if float((candidate > limits).mean()) <= epsilon:
                for k in members:
                    picks[episode[k]] = to_model[k]
                running = candidate
        start = end
    return picks


def take_within_budget(
    segments: Segments,
    n_episodes: int,
    budget_extra: float,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """ROI 내림차순으로 구간을 집어 예산 여유를 소진한다.

    ``budget_extra``는 all-light 대비 **추가로 쓸 수 있는** 비용이다.
    반환값은 문항별 선택 모델 인덱스 ``picks[n_episodes]``.
    """

    picks = segments.base_model.copy()
    if len(segments.order) == 0 or budget_extra <= 0:
        return picks

    idx = segments.order
    grp = segments.group
    keep = np.ones(len(idx), dtype=bool) if mask is None else mask[idx]

    episode = segments.episode
    from_model = segments.from_model
    to_model = segments.to_model
    delta_cost = segments.delta_cost

    # 예산에 안 들어가는 그룹은 **건너뛰고** 더 싼 그룹으로 계속 채운다.
    # 첫 초과에서 멈추면 남은 예산이 통째로 버려진다.
    # 건너뛰더라도 선행 구간이 집혔을 때만 집으므로 포락선 순서는 유지된다.
    remaining = float(budget_extra)
    start = 0
    n_seg = len(idx)
    while start < n_seg:
        end = start + 1
        while end < n_seg and grp[end] == grp[start]:
            end += 1

        members = [
            k
            for k in idx[start:end][keep[start:end]]
            if picks[episode[k]] == from_model[k]
        ]
        if members:
            need = math.fsum(float(delta_cost[k]) for k in members)
            if need <= remaining:
                for k in members:
                    picks[episode[k]] = to_model[k]
                remaining -= need
        start = end
    return picks
