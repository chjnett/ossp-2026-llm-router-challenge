# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""파산 게이트 — 분포가 흔들려도 예산을 넘기지 않는지 본다.

공식 baseline은 공개 Dev에서 Premium 비용 비율 3.985로 통과했는데 채점셋에서
약 4.2가 나와 Premium 0점을 받았다. Dev 한 번의 통과는 아무것도 보장하지 않는다.

여기서는 **예측으로 배분하고 실제 비용으로 채점한다.** 그게 평가에서 벌어지는
일이다. 라우터는 자기가 얼마를 쓰는지 끝까지 모른다.

주의: 시뮬레이터는 '공개 계열 8종의 재가중' 범위만 흔든다. 공개셋에 없는 계열이
비공개 평가에 나오면 시행 횟수를 아무리 늘려도 막지 못한다 (DESIGN.md §6 미검증 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np

from .allocate import allocate
from .data import TIERS, Dataset

Sampler = Callable[[np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class TierStress:
    tier: str
    trials: int
    failures: int
    ratio_over_limit: np.ndarray  # 실제비율 / 한도. 1.0을 넘으면 파산

    @property
    def failure_rate(self) -> float:
        return self.failures / max(1, self.trials)

    def quantiles(self) -> tuple[float, float, float, float]:
        q = np.percentile(self.ratio_over_limit, [50, 95, 99, 100])
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    def __str__(self) -> str:
        p50, p95, p99, mx = self.quantiles()
        flag = "  ✗" if self.failures else ""
        return (
            f"{self.tier:9s} 파산 {self.failures:5d}/{self.trials:<5d} "
            f"({self.failure_rate:6.2%})  사용률 p50 {p50:.3f} p95 {p95:.3f} "
            f"p99 {p99:.3f} max {mx:.3f}{flag}"
        )


@dataclass(frozen=True)
class GateResult:
    scenario: str
    tiers: Dict[str, TierStress]

    @property
    def passed(self) -> bool:
        return all(t.failures == 0 for t in self.tiers.values())

    def __str__(self) -> str:
        head = f"[{self.scenario}]  {'통과' if self.passed else '실패'}"
        rows = "\n".join(f"    {self.tiers[t]}" for t in TIERS)
        return f"{head}\n{rows}"


def family_reweight_sampler(
    family: np.ndarray, size: int, concentration: float
) -> Sampler:
    """계열 비중을 Dirichlet으로 흔들어 재표집한다.

    ``concentration``이 작을수록 한 계열이 배치를 지배한다. 5.0은 온건한 이동,
    0.7은 극단적 이동이다.
    """

    labels = np.unique(family)
    pools = [np.where(family == f)[0] for f in labels]

    def sample(rng: np.random.Generator) -> np.ndarray:
        weights = rng.dirichlet(np.full(len(labels), concentration))
        counts = rng.multinomial(size, weights)
        parts = [
            rng.choice(pool, size=c, replace=True)
            for pool, c in zip(pools, counts)
            if c > 0 and len(pool) > 0
        ]
        return np.concatenate(parts) if parts else pools[0][:1]

    return sample


def family_dominant_sampler(
    family: np.ndarray, size: int, share: float = 0.75
) -> Sampler:
    """한 계열이 배치의 대부분을 차지하는 최악의 경우.

    비공개 평가가 특정 과제 유형에 쏠렸을 때를 흉내 낸다. 계열마다 돌아가며
    지배자가 되므로 leave-one-family-out의 역방향 검사이기도 하다.
    """

    labels = np.unique(family)
    pools = [np.where(family == f)[0] for f in labels]
    other = np.arange(len(family))

    def sample(rng: np.random.Generator) -> np.ndarray:
        pick = int(rng.integers(len(labels)))
        n_main = int(size * share)
        main = rng.choice(pools[pick], size=n_main, replace=True)
        rest = rng.choice(other, size=size - n_main, replace=True)
        return np.concatenate([main, rest])

    return sample


def uniform_sampler(n: int, size: int) -> Sampler:
    def sample(rng: np.random.Generator) -> np.ndarray:
        return rng.choice(n, size=size, replace=True)

    return sample


def run_scenario(
    dataset: Dataset,
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    sampler: Sampler,
    *,
    scenario: str,
    util: Dict[str, float] | float,
    multipliers: Dict[str, float],
    allow: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    mu: float = 0.0,
    trials: int = 2000,
    seed: int = 0,
    size_penalty: float = 0.0,
    headroom: Dict[str, float] | None = None,
) -> GateResult:
    """한 시나리오를 ``trials``회 돌려 등급별 파산 횟수를 센다.

    ``size_penalty``는 배치 크기에 따른 사용률 감쇠 계수다. 라우터가 실행
    시점에 적용하는 것과 같은 규칙을 여기서도 적용해야 게이트가 실제 동작을
    잰다.
    """

    rng = np.random.default_rng(seed)
    base_utils = util if isinstance(util, dict) else {t: util for t in TIERS}
    keys = np.asarray(dataset.keys)

    ratios = {t: np.empty(trials, dtype=float) for t in TIERS}
    fails = {t: 0 for t in TIERS}

    for trial in range(trials):
        idx = sampler(rng)
        s_i, c_i = s_hat[idx], c_hat[idx]
        allow_i = None if allow is None else allow[idx]
        sd_i = None if sd is None else sd[idx]
        keys_i = keys[idx].tolist()
        true_cost = dataset.cost[idx]
        # 한도의 분모는 실제 light 비용이다. 라우터는 이 값을 모른다.
        light = float(true_cost[:, 0].sum())
        shrink = size_penalty / max(1.0, float(len(idx))) ** 0.5
        if headroom is None:
            utils = {t: max(0.0, u - shrink) for t, u in base_utils.items()}
        else:
            # 여윳돈 기준. 라우터가 실행 시점에 쓰는 규칙과 같아야 한다.
            utils = {
                t: (1.0 + max(0.0, h - shrink) * (multipliers[t] - 1.0)) / multipliers[t]
                for t, h in headroom.items()
            }

        for tier in TIERS:
            picks = allocate(
                s_i,
                c_i,
                multiplier=multipliers[tier],
                util=utils[tier],
                allow=allow_i,
                sd=sd_i,
                mu=mu,
                keys=keys_i,
            ).picks
            used = float(true_cost[np.arange(len(idx)), picks].sum())
            over = (used / light) / multipliers[tier]
            ratios[tier][trial] = over
            if over > 1.0:
                fails[tier] += 1

    return GateResult(
        scenario=scenario,
        tiers={
            t: TierStress(t, trials, fails[t], ratios[t]) for t in TIERS
        },
    )


def default_scenarios(
    family: np.ndarray, n: int, *, sizes: Sequence[int] = (200, 800, 2640)
) -> Dict[str, Sampler]:
    """게이트 기본 구성. 크기·온건 이동·극단 이동·계열 지배를 함께 본다."""

    scenarios: Dict[str, Sampler] = {}
    for size in sizes:
        scenarios[f"size-{size}"] = uniform_sampler(n, size)
    scenarios["shift-mild(conc=5)"] = family_reweight_sampler(family, 800, 5.0)
    scenarios["shift-extreme(conc=0.7)"] = family_reweight_sampler(family, 800, 0.7)
    scenarios["family-dominant(75%)"] = family_dominant_sampler(family, 800, 0.75)
    return scenarios


def run_gate(
    dataset: Dataset,
    s_hat: np.ndarray,
    c_hat: np.ndarray,
    *,
    family: np.ndarray,
    util: Dict[str, float] | float,
    multipliers: Dict[str, float],
    allow: np.ndarray | None = None,
    sd: np.ndarray | None = None,
    mu: float = 0.0,
    trials: int = 2000,
    seed: int = 0,
    scenarios: Dict[str, Sampler] | None = None,
    size_penalty: float = 0.0,
    headroom: Dict[str, float] | None = None,
) -> list[GateResult]:
    """모든 시나리오를 돌린다. 하나라도 파산이 있으면 게이트 불통과다."""

    scenarios = scenarios or default_scenarios(family, len(dataset))
    results = []
    for offset, (name, sampler) in enumerate(scenarios.items()):
        results.append(
            run_scenario(
                dataset,
                s_hat,
                c_hat,
                sampler,
                scenario=name,
                util=util,
                multipliers=multipliers,
                allow=allow,
                sd=sd,
                mu=mu,
                trials=trials,
                seed=seed + offset * 1009,
                size_penalty=size_penalty,
                headroom=headroom,
            )
        )
    return results


def gate_passed(results: Sequence[GateResult]) -> bool:
    return all(r.passed for r in results)
