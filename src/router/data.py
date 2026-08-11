# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""공개 Train/Dev 자료를 실험용 배열로 올리는 계층.

라우터 런타임은 이 모듈을 쓰지 않는다. 오프라인 학습·평가 전용이다.
문항 ID와 입력 순서는 여기서만 다루고, 특징 추출과 모델 선택에는 넘기지 않는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ossp_router.protocol import (
    Episode,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
)

# 싼 것부터. argmax 동률을 이 순서로 결정적으로 깬다 (RULES B3).
MODEL_IDS: Tuple[str, ...] = ("ax31-light", "ax31", "axk1-think")
LIGHT = MODEL_IDS[0]
TIERS: Tuple[str, ...] = ("fast", "balanced", "premium")

REPO_ROOT = Path(__file__).resolve().parents[2]


def episode_text(episode: Episode) -> str:
    """라우팅 시점에 볼 수 있는 텍스트만 반환한다."""

    if episode.prompt is not None:
        return episode.prompt
    assert episode.messages is not None
    return "\n".join(message.content for message in episode.messages)


def content_key(text: str) -> str:
    """프롬프트 내용에서만 유도한 고정 키.

    내장 ``hash()``는 PYTHONHASHSEED에 따라 프로세스마다 달라지므로 쓰지 않는다
    (RULES B4). 정렬·fold 배정에 쓰는 유일한 키다.
    """

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class Dataset:
    """한 split의 프롬프트와 모델별 결과를 함께 들고 있는 실험용 묶음."""

    split: str
    challenge_id: str
    inputs: InputBatch
    outcomes: OutcomeBatch
    policy: RoutingPolicy
    texts: Tuple[str, ...]
    episode_ids: Tuple[str, ...]
    keys: Tuple[str, ...]
    score: np.ndarray          # [n, 3] 실제 관측 점수
    input_tokens: np.ndarray   # [n, 3]
    output_tokens: np.ndarray  # [n, 3]
    generations: np.ndarray    # [n]
    cost: np.ndarray           # [n, 3] 공개 정책으로 계산한 실제 비용

    def __len__(self) -> int:
        return len(self.texts)

    @property
    def light_baseline_cost(self) -> float:
        return float(self.cost[:, 0].sum())

    def subset(self, index: Sequence[int]) -> "Dataset":
        idx = np.asarray(index, dtype=int)
        keep = {self.episode_ids[i] for i in idx}
        # inputs/outcomes도 같이 잘라 둔다. 부분집합 Dataset을 그대로 채점기에
        # 넘겼을 때 문항 수가 어긋나는 사고를 막는다.
        inputs = InputBatch(
            schema_version=self.inputs.schema_version,
            challenge_id=self.inputs.challenge_id,
            split=self.inputs.split,
            episodes=tuple(e for e in self.inputs.episodes if e.episode_id in keep),
        )
        outcomes = OutcomeBatch(
            schema_version=self.outcomes.schema_version,
            challenge_id=self.outcomes.challenge_id,
            split=self.outcomes.split,
            outcomes=tuple(o for o in self.outcomes.outcomes if o.episode_id in keep),
        )
        return Dataset(
            split=self.split,
            challenge_id=self.challenge_id,
            inputs=inputs,
            outcomes=outcomes,
            policy=self.policy,
            texts=tuple(self.texts[i] for i in idx),
            episode_ids=tuple(self.episode_ids[i] for i in idx),
            keys=tuple(self.keys[i] for i in idx),
            score=self.score[idx],
            input_tokens=self.input_tokens[idx],
            output_tokens=self.output_tokens[idx],
            generations=self.generations[idx],
            cost=self.cost[idx],
        )

    def folds(self, k: int = 5) -> List[np.ndarray]:
        """콘텐츠 해시로 고정한 fold. 순서·ID가 바뀌어도 같은 분할이 나온다."""

        bucket = np.array(
            [int(key[:8], 16) % k for key in self.keys], dtype=int
        )
        return [np.where(bucket == f)[0] for f in range(k)]


def _rate_matrix(policy: RoutingPolicy) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    fixed = np.array(
        [float(policy.models[m].fixed_cost) for m in MODEL_IDS], dtype=float
    )
    rate_in = np.array(
        [float(policy.models[m].input_token_rate) for m in MODEL_IDS], dtype=float
    )
    rate_out = np.array(
        [float(policy.models[m].output_token_rate) for m in MODEL_IDS], dtype=float
    )
    return fixed, rate_in, rate_out, policy.token_unit


def cost_from_tokens(
    input_tokens: np.ndarray,
    output_tokens: np.ndarray,
    policy: RoutingPolicy,
) -> np.ndarray:
    """공개 비용 정책을 그대로 적용한다. 계수를 코드에 박지 않는다."""

    fixed, rate_in, rate_out, unit = _rate_matrix(policy)
    return fixed + (input_tokens * rate_in + output_tokens * rate_out) / unit


def load_dataset(
    split: str,
    *,
    root: Path | None = None,
    policy: RoutingPolicy | None = None,
) -> Dataset:
    """``data/materialized/<split>/inputs.json``과 ``data/<split>/outcomes.json``을 읽는다."""

    root = root or REPO_ROOT
    policy = policy or load_bundled_policy()
    inputs = load_input(root / "data" / "materialized" / split / "inputs.json")
    outcomes = load_outcomes(root / "data" / split / "outcomes.json")

    # OutcomeBatch.outcomes 는 (episode_id, model_id) 단위로 평탄화된 튜플이다.
    by_key = {(o.episode_id, o.model_id): o for o in outcomes.outcomes}
    missing = [
        e.episode_id
        for e in inputs.episodes
        if any((e.episode_id, m) not in by_key for m in MODEL_IDS)
    ]
    if missing:
        raise ValueError(f"{split}: outcome이 없는 문항 {len(missing)}개 (예: {missing[:3]})")

    texts, ids, keys = [], [], []
    n = len(inputs.episodes)
    score = np.zeros((n, 3), dtype=float)
    tok_in = np.zeros((n, 3), dtype=float)
    tok_out = np.zeros((n, 3), dtype=float)
    gens = np.zeros(n, dtype=int)

    for i, episode in enumerate(inputs.episodes):
        text = episode_text(episode)
        texts.append(text)
        ids.append(episode.episode_id)
        keys.append(content_key(text))
        for j, model_id in enumerate(MODEL_IDS):
            outcome = by_key[(episode.episode_id, model_id)]
            score[i, j] = float(outcome.score)
            tok_in[i, j] = outcome.input_tokens
            tok_out[i, j] = outcome.output_tokens
        gens[i] = by_key[(episode.episode_id, LIGHT)].num_generations

    return Dataset(
        split=split,
        challenge_id=inputs.challenge_id,
        inputs=inputs,
        outcomes=outcomes,
        policy=policy,
        texts=tuple(texts),
        episode_ids=tuple(ids),
        keys=tuple(keys),
        score=score,
        input_tokens=tok_in,
        output_tokens=tok_out,
        generations=gens,
        cost=cost_from_tokens(tok_in, tok_out, policy),
    )


def budget_multipliers(policy: RoutingPolicy) -> Dict[str, float]:
    return {t: float(policy.tiers[t].budget_multiplier) for t in TIERS}


def tier_weights(policy: RoutingPolicy) -> Dict[str, Decimal]:
    return {t: policy.tiers[t].weight for t in TIERS}
