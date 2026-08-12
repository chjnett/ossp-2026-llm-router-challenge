# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""오프라인 평가 하네스.

점수는 **항상 공식 채점기**(``ossp_router.scoring.score_submissions``)로 낸다.
자체 구현한 지름길 점수를 최종 지표로 쓰지 않는다 — 그 순간 실험 결과를 믿을 수 없다.
numpy는 λ 탐색 내부에서만 쓰고, 판정은 Decimal 경로가 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Mapping, Sequence

import numpy as np

from ossp_router.protocol import (
    Decision,
    InputBatch,
    OutcomeBatch,
    Submission,
    parse_submission,
    submission_to_dict,
)
from ossp_router.scoring import score_submissions

from .data import MODEL_IDS, TIERS, Dataset


@dataclass(frozen=True)
class TierResult:
    tier: str
    score: Decimal
    budget_ratio: Decimal
    budget_multiplier: Decimal
    passed: bool
    near_budget: bool
    counts: Dict[str, int]

    @property
    def headroom(self) -> Decimal:
        """한도 대비 남은 여유. 음수면 초과다."""

        return self.budget_multiplier - self.budget_ratio

    def __str__(self) -> str:
        mark = "" if self.passed else "  ✗예산초과"
        warn = " ~한도근접" if self.near_budget and self.passed else ""
        return (
            f"{self.tier:8s} {float(self.score):.4f}  "
            f"ratio {float(self.budget_ratio):.3f}/{float(self.budget_multiplier):.2f}  "
            f"여유 {float(self.headroom):+.3f}{warn}{mark}"
        )


@dataclass(frozen=True)
class Evaluation:
    tiers: Dict[str, TierResult]
    final_score: Decimal
    all_passed: bool

    def __str__(self) -> str:
        rows = "\n".join(f"  {self.tiers[t]}" for t in TIERS)
        return f"{rows}\n  가중 최종 {float(self.final_score):.4f}"

    def as_record(self) -> Dict[str, object]:
        return {
            "final_score": str(self.final_score),
            "all_passed": self.all_passed,
            "tiers": {
                t: {
                    "score": str(r.score),
                    "budget_ratio": str(r.budget_ratio),
                    "headroom": str(r.headroom),
                    "passed": r.passed,
                    "near_budget": r.near_budget,
                    "counts": r.counts,
                }
                for t, r in self.tiers.items()
            },
        }


def _sub_batches(dataset: Dataset, index: Sequence[int]) -> tuple[InputBatch, OutcomeBatch]:
    """CV fold처럼 부분집합을 공식 채점기에 넘기기 위한 배치를 만든다."""

    keep_ids = {dataset.episode_ids[i] for i in index}
    episodes = tuple(e for e in dataset.inputs.episodes if e.episode_id in keep_ids)
    outcomes = tuple(o for o in dataset.outcomes.outcomes if o.episode_id in keep_ids)
    inputs = InputBatch(
        schema_version=dataset.inputs.schema_version,
        challenge_id=dataset.inputs.challenge_id,
        split=dataset.inputs.split,
        episodes=episodes,
    )
    outcome_batch = OutcomeBatch(
        schema_version=dataset.outcomes.schema_version,
        challenge_id=dataset.outcomes.challenge_id,
        split=dataset.outcomes.split,
        outcomes=outcomes,
    )
    return inputs, outcome_batch


def build_submission(
    inputs: InputBatch, policy_id: str, tier: str, picks: np.ndarray
) -> Submission:
    """picks[i]는 inputs.episodes[i]에 대한 MODEL_IDS 인덱스다."""

    if len(picks) != len(inputs.episodes):
        raise ValueError(f"picks {len(picks)}개 != 문항 {len(inputs.episodes)}개")
    decisions = tuple(
        Decision(episode.episode_id, MODEL_IDS[int(picks[i])])
        for i, episode in enumerate(inputs.episodes)
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy_id,
        split=inputs.split,
        tier=tier,
        decisions=decisions,
    )
    # 생성기와 공식 파서를 같은 엄격 경로에 태운다.
    return parse_submission(submission_to_dict(submission))


def evaluate(
    dataset: Dataset,
    picks_by_tier: Mapping[str, np.ndarray],
    *,
    index: Sequence[int] | None = None,
) -> Evaluation:
    """세 등급의 선택을 공식 채점기로 채점한다."""

    if set(picks_by_tier) != set(TIERS):
        raise ValueError(f"세 등급이 모두 필요하다: {sorted(picks_by_tier)}")

    if index is None:
        inputs, outcomes = dataset.inputs, dataset.outcomes
    else:
        inputs, outcomes = _sub_batches(dataset, index)

    submissions = [
        build_submission(inputs, dataset.policy.policy_id, tier, picks_by_tier[tier])
        for tier in TIERS
    ]
    report = score_submissions(inputs, outcomes, submissions, dataset.policy)

    tiers: Dict[str, TierResult] = {}
    for tier in TIERS:
        row = report["tiers"][tier]
        tiers[tier] = TierResult(
            tier=tier,
            score=Decimal(row["quality_score"]),
            budget_ratio=Decimal(row["budget_ratio"]),
            budget_multiplier=Decimal(row["budget_multiplier"]),
            passed=bool(row["budget_passed"]),
            near_budget=bool(row["near_budget"]),
            counts=dict(row["model_counts"]),
        )
    return Evaluation(
        tiers=tiers,
        final_score=Decimal(report["final_score"]),
        all_passed=all(t.passed for t in tiers.values()),
    )


def evaluate_cv(
    dataset: Dataset,
    pick_fn,
    *,
    k: int = 5,
) -> tuple[Evaluation, list[Evaluation]]:
    """fold별로 ``pick_fn(train_idx, test_idx) -> {tier: picks}``를 부르고 합쳐서 채점한다.

    반환값은 (전체 out-of-fold 평가, fold별 평가)다.
    전체 평가는 fold별 예측을 원래 자리에 되돌린 뒤 한 번에 채점한 값이라
    fold 크기 차이에 영향을 받지 않는다.
    """

    folds = dataset.folds(k)
    n = len(dataset)
    oof = {tier: np.zeros(n, dtype=int) for tier in TIERS}
    per_fold: list[Evaluation] = []

    for f, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[g] for g in range(k) if g != f])
        picks = pick_fn(train_idx, test_idx)
        for tier in TIERS:
            oof[tier][test_idx] = picks[tier]
        per_fold.append(evaluate(dataset, picks, index=test_idx))

    return evaluate(dataset, oof), per_fold
