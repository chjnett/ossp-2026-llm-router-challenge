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

import hashlib
import re
from typing import Callable, Dict, Protocol, Sequence

import numpy as np

from .data import MODEL_IDS, TIERS, Dataset, cost_from_tokens
from .features import FEATURE_NAMES, FAMILIES, extract, family_codes
from .hash_features import FEATURE_VERSION as HASH_FEATURE_VERSION
from .hash_features import extract_hash_features

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


def _hash_ridge_fit(design: np.ndarray, target: np.ndarray, alpha: float) -> tuple:
    """공식 베이스라인과 같은 표준화 ridge를 다중 목표에 한 번에 적합한다."""

    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    mean = design.mean(axis=0)
    scale = design.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design - mean) / scale
    intercept = target.mean(axis=0)
    centered = target - intercept
    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + alpha * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        system = standardized.T @ standardized + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, standardized.T @ centered)
    return mean, scale, intercept, coefficients


def _hash_ridge_apply(design: np.ndarray, fitted: tuple) -> np.ndarray:
    mean, scale, intercept, coefficients = fitted
    return (design - mean) / scale @ coefficients + intercept


def _hash_ridge_oof_apply(
    design: np.ndarray,
    target: np.ndarray,
    folds: Sequence[np.ndarray],
    alpha: float,
) -> np.ndarray:
    """각 행을 적합에서 제외한 ridge 예측을 원래 행 순서로 반환한다."""

    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    prediction = np.full(target.shape, np.nan, dtype=float)
    all_rows = np.arange(len(design))
    coverage = np.zeros(len(design), dtype=int)
    for held_out in folds:
        held_out = np.asarray(held_out, dtype=int)
        if len(held_out) == 0:
            continue
        train_rows = all_rows[~np.isin(all_rows, held_out)]
        if len(train_rows) == 0:
            raise ValueError("OOF ridge에는 비어 있지 않은 학습 fold가 필요하다")
        fitted = _hash_ridge_fit(design[train_rows], target[train_rows], alpha)
        prediction[held_out] = _hash_ridge_apply(design[held_out], fitted)
        coverage[held_out] += 1
    if not np.all(coverage == 1):
        raise ValueError("OOF fold가 모든 학습 행을 정확히 포함해야 한다")
    return prediction


def _pack_hash_ridge(fitted: tuple) -> list:
    return [np.asarray(value).tolist() for value in fitted]


def _unpack_hash_ridge(state: list) -> tuple:
    return tuple(np.asarray(value, dtype=float) for value in state)


@register(SCORE_HEADS, "hash_ridge")
class HashRidgeScore:
    """전역 signed unigram/bigram ridge 점수 예측.

    계열 평균이 버리던 계열 내부의 lexical 신호를 살린다. 공개 베이스라인의
    표현은 재사용하되, 위험했던 고정 safety ratio 선택기는 쓰지 않고 우리
    배치 적응형 예산 할당기로 넘긴다.
    """

    def __init__(self, alpha: float = 100.0, bins: int = 256) -> None:
        self.alpha = float(alpha)
        self.bins = int(bins)
        extract_hash_features((), self.bins)
        self.version = (
            f"hashscore.v1(f={HASH_FEATURE_VERSION},a={self.alpha:g},b={self.bins})"
        )

    def fit(self, train: Dataset) -> None:
        design = extract_hash_features(train.texts, self.bins)
        self._fitted = _hash_ridge_fit(design, train.score, self.alpha)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        design = extract_hash_features(texts, self.bins)
        return np.clip(_hash_ridge_apply(design, self._fitted), 0.0, 1.0)

    def state(self) -> dict:
        return {"fitted": _pack_hash_ridge(self._fitted)}

    def load_state(self, state: dict) -> None:
        self._fitted = _unpack_hash_ridge(state["fitted"])


def _hash_ridge_fit_weighted(
    design: np.ndarray, target: np.ndarray, alpha: float, weights: np.ndarray
) -> tuple:
    """역분산 가중 ridge.

    score는 num_generations(2 또는 4)회 생성의 평균이라 문항마다 노이즈
    분산이 다르다(DESIGN §2.6). 노이즈가 작은(생성 수가 많은) 행에 더 큰
    비중을 두는 가중 최소제곱이다. 가중치는 generations에 비례시킨다.
    """

    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.mean()
    w = weights[:, None]
    sw = float(weights.sum())
    mean = (design * w).sum(axis=0) / sw
    centered_x = design - mean
    var = (weights[:, None] * centered_x * centered_x).sum(axis=0) / sw
    scale = np.sqrt(var)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = centered_x / scale
    intercept = (target * w).sum(axis=0) / sw
    centered = target - intercept
    rows, columns = standardized.shape
    if rows <= columns:
        raise ValueError("가중 ridge는 rows > columns 전용이다")
    xtw = standardized.T * weights  # [columns, rows]
    system = xtw @ standardized + alpha * np.eye(columns)
    coefficients = np.linalg.solve(system, xtw @ centered)
    return mean, scale, intercept, coefficients


@register(SCORE_HEADS, "hash_ridge_weighted")
class HashRidgeWeightedScore:
    """num_generations로 역분산 가중한 hashed ridge 점수 예측.

    HashRidgeScore와 표현·계약이 같고, score 노이즈가 작은(생성 수가 많은)
    문항에 더 큰 비중을 둔다. 노이즈 상한을 직접 공략하는 마지막 점수 헤드
    시도.
    """

    def __init__(self, alpha: float = 100.0, bins: int = 256) -> None:
        self.alpha = float(alpha)
        self.bins = int(bins)
        extract_hash_features((), self.bins)
        self.version = (
            f"hashscore-weighted.v1(f={HASH_FEATURE_VERSION},a={self.alpha:g},b={self.bins})"
        )

    def fit(self, train: Dataset) -> None:
        design = extract_hash_features(train.texts, self.bins)
        weights = np.asarray(train.generations, dtype=float)
        self._fitted = _hash_ridge_fit_weighted(
            design, train.score, self.alpha, weights
        )

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        design = extract_hash_features(texts, self.bins)
        return np.clip(_hash_ridge_apply(design, self._fitted), 0.0, 1.0)

    def state(self) -> dict:
        return {"fitted": _pack_hash_ridge(self._fitted)}

    def load_state(self, state: dict) -> None:
        self._fitted = _unpack_hash_ridge(state["fitted"])


# --- 임베딩 점수 헤드 (Track B) -------------------------------------------
# fastembed는 오프라인 학습·평가 전용이다. 런타임 이미지에 넣으려면 별도로
# 모델 파일과 추론 경로(onnxruntime 또는 numpy)를 고정하고 A3 출처·라이선스를
# 기록해야 한다. 여기서는 "임베딩이 라우팅 OOF를 올리는가"를 먼저 판정한다.

_EMBED_MODEL = None
_EMBED_CACHE: Dict[str, np.ndarray] = {}
_EMBED_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _embed(texts: Sequence[str]) -> np.ndarray:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from fastembed import TextEmbedding

        _EMBED_MODEL = TextEmbedding(model_name=_EMBED_NAME)
    missing = [t for t in texts if t not in _EMBED_CACHE]
    if missing:
        for text, vec in zip(missing, _EMBED_MODEL.embed(missing)):
            _EMBED_CACHE[text] = np.asarray(vec, dtype=np.float64)
    return np.stack([_EMBED_CACHE[t] for t in texts])


@register(SCORE_HEADS, "embedding_ridge")
class EmbeddingRidgeScore:
    """사전학습 다국어 임베딩 + ridge 점수 예측.

    해시 n-gram이 못 잡는 의미적 난이도를 사전학습 인코더로 표현한다. Track B의
    판정용 구현이다. 결정성: ONNX float32 고정, mean-pooling, dropout 없음.
    """

    def __init__(self, alpha: float = 1000.0) -> None:
        self.alpha = float(alpha)
        self.version = f"embedscore.v1({_EMBED_NAME.rsplit('/', 1)[-1]},a={self.alpha:g})"

    def fit(self, train: Dataset) -> None:
        design = _embed(train.texts)
        self._fitted = _hash_ridge_fit(design, train.score, self.alpha)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        design = _embed(texts)
        return np.clip(_hash_ridge_apply(design, self._fitted), 0.0, 1.0)

    def state(self) -> dict:
        return {"fitted": _pack_hash_ridge(self._fitted)}

    def load_state(self, state: dict) -> None:
        self._fitted = _unpack_hash_ridge(state["fitted"])


@register(SCORE_HEADS, "family_hash_ridge")
class FamilyHashRidgeScore:
    """선택한 prompt family마다 독립적인 hashed ridge를 적합한다.

    전역 ridge는 서로 다른 계열의 lexical 계수를 하나로 평균낸다. 특정 계열의
    신호가 반대 방향이면 강한 전역 축소만으로는 순위를 복원할 수 없다. 충분한
    표본이 있는 사전 지정 계열에는 로컬 계수를 쓰고, 그 밖에는 전역 적합으로
    폴백한다.
    """

    def __init__(
        self,
        alpha: float = 1000.0,
        bins: int = 256,
        active_families: Sequence[str] = (),
        min_family: int = 40,
    ) -> None:
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.min_family = int(min_family)
        unknown = set(active_families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        if not active_families:
            raise ValueError("family_hash_ridge active_families는 비어 있으면 안 된다")
        self._active_codes = frozenset(
            i for i, name in enumerate(FAMILIES) if name in active_families
        )
        extract_hash_features((), self.bins)
        scope = "+".join(sorted(active_families))
        self.version = (
            f"familyhashscore.v1(f={HASH_FEATURE_VERSION},a={self.alpha:g},"
            f"b={self.bins},min={self.min_family},scope={scope})"
        )

    def fit(self, train: Dataset) -> None:
        design = extract_hash_features(train.texts, self.bins)
        self._global = _hash_ridge_fit(design, train.score, self.alpha)
        codes = family_codes(train.texts)
        self._by_family: Dict[int, tuple] = {}
        for code in self._active_codes:
            mask = codes == code
            if mask.sum() >= self.min_family:
                self._by_family[code] = _hash_ridge_fit(
                    design[mask], train.score[mask], self.alpha
                )

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        design = extract_hash_features(texts, self.bins)
        out = _hash_ridge_apply(design, self._global)
        codes = family_codes(texts)
        for code, fitted in self._by_family.items():
            mask = codes == code
            if mask.any():
                out[mask] = _hash_ridge_apply(design[mask], fitted)
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "global": _pack_hash_ridge(self._global),
            "by_family": {
                str(code): _pack_hash_ridge(fitted)
                for code, fitted in self._by_family.items()
            },
        }

    def load_state(self, state: dict) -> None:
        self._global = _unpack_hash_ridge(state["global"])
        self._by_family = {
            int(code): _unpack_hash_ridge(fitted)
            for code, fitted in state["by_family"].items()
        }


@register(SCORE_HEADS, "hash_response")
class HashResponseScore:
    """hashed lexical 특징으로 두 승격 단계의 양성확률을 예측한다.

    score 자체를 회귀하면 대부분 0인 이득이 평균으로 수축돼 ax31을 넓게
    뿌린다. 대신 light→ax31, ax31→K1 이득이 양수인지 각각 예측하고 family별
    양성/비양성 조건부 이득을 혼합한다. light 점수는 별도 강축소 hash ridge의
    예측을 유지한다.
    """

    def __init__(
        self,
        alpha: float = 1000.0,
        bins: int = 128,
        strength: float = 1.0,
        base_alpha: float = 32000.0,
        active_families: Sequence[str] = (),
        min_family: int = 40,
    ) -> None:
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.strength = float(strength)
        self.base_alpha = float(base_alpha)
        self.min_family = int(min_family)
        unknown = set(active_families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        self._active_codes = frozenset(
            i for i, name in enumerate(FAMILIES) if name in active_families
        )
        extract_hash_features((), self.bins)
        scope = "+".join(sorted(active_families)) or "global"
        self._base = HashRidgeScore(alpha=self.base_alpha, bins=self.bins)
        self.version = (
            f"hashresponse.v1(f={HASH_FEATURE_VERSION},a={self.alpha:g},"
            f"b={self.bins},s={self.strength:g},ba={self.base_alpha:g},"
            f"min={self.min_family},scope={scope})"
        )

    def fit(self, train: Dataset) -> None:
        self._base.fit(train)
        design = extract_hash_features(train.texts, self.bins)
        codes = family_codes(train.texts)
        score = np.asarray(train.score, dtype=float)
        gain = np.column_stack(
            [score[:, 1] - score[:, 0], score[:, 2] - score[:, 1]]
        )
        positive = (gain > 0).astype(float)
        self._probability = _hash_ridge_fit(design, positive, self.alpha)
        self._probability_by_family: Dict[int, tuple] = {}
        for code in self._active_codes:
            mask = codes == code
            if mask.sum() >= self.min_family:
                self._probability_by_family[code] = _hash_ridge_fit(
                    design[mask], positive[mask], self.alpha
                )

        self._share = np.zeros((len(FAMILIES), 2))
        self._hit = np.zeros((len(FAMILIES), 2))
        self._miss = np.zeros((len(FAMILIES), 2))
        for code in range(len(FAMILIES)):
            family_mask = codes == code
            source = family_mask if family_mask.any() else np.ones(len(train), dtype=bool)
            for stage in range(2):
                values = gain[source, stage]
                stage_positive = values > 0
                self._share[code, stage] = float(stage_positive.mean())
                self._hit[code, stage] = (
                    float(values[stage_positive].mean())
                    if stage_positive.any()
                    else 0.0
                )
                self._miss[code, stage] = (
                    float(values[~stage_positive].mean())
                    if (~stage_positive).any()
                    else 0.0
                )

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        design = extract_hash_features(texts, self.bins)
        codes = family_codes(texts)
        probability = _hash_ridge_apply(design, self._probability)
        for code, fitted in self._probability_by_family.items():
            mask = codes == code
            if mask.any():
                probability[mask] = _hash_ridge_apply(design[mask], fitted)
        probability = self._share[codes] + self.strength * (
            probability - self._share[codes]
        )
        probability = np.clip(probability, 0.0, 1.0)
        stage_gain = (
            probability * self._hit[codes]
            + (1.0 - probability) * self._miss[codes]
        )
        light = self._base.predict(texts)[:, 0]
        out = np.column_stack(
            [light, light + stage_gain[:, 0], light + stage_gain.sum(axis=1)]
        )
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "base": self._base.state(),
            "probability": _pack_hash_ridge(self._probability),
            "probability_by_family": {
                str(code): _pack_hash_ridge(fitted)
                for code, fitted in self._probability_by_family.items()
            },
            "share": self._share.tolist(),
            "hit": self._hit.tolist(),
            "miss": self._miss.tolist(),
        }

    def load_state(self, state: dict) -> None:
        self._base.load_state(state["base"])
        self._probability = _unpack_hash_ridge(state["probability"])
        self._probability_by_family = {
            int(code): _unpack_hash_ridge(fitted)
            for code, fitted in state["probability_by_family"].items()
        }
        self._share = np.asarray(state["share"], dtype=float)
        self._hit = np.asarray(state["hit"], dtype=float)
        self._miss = np.asarray(state["miss"], dtype=float)


@register(SCORE_HEADS, "hash_knn")
class HashKNNScore:
    """프롬프트 유사 이웃의 공개 outcome을 강한 전역 예측에 축소한다.

    정규화 signed unigram/bigram의 cosine 유사도를 쓰며, 기본적으로 같은
    prompt family 안에서만 이웃을 찾는다. 원문·문항 ID·출처는 아티팩트에
    저장하지 않는다. 학습 행은 prompt 내용 해시로 정렬해 입력 순서와 top-k
    동률 처리의 영향을 제거한다.

    ``prior``는 전역 hash ridge 예측에 주는 의사 이웃 질량이다. 유사 이웃이
    없으면 정확히 전역 예측으로 폴백하고, 낮은 유사도는 ``threshold``로
    제거한다.
    """

    def __init__(
        self,
        bins: int = 64,
        neighbors: int = 16,
        prior: float = 8.0,
        power: float = 2.0,
        threshold: float = 0.0,
        base_alpha: float = 32000.0,
        same_family: bool = True,
    ) -> None:
        self.bins = int(bins)
        self.neighbors = int(neighbors)
        self.prior = float(prior)
        self.power = float(power)
        self.threshold = float(threshold)
        self.base_alpha = float(base_alpha)
        self.same_family = bool(same_family)
        extract_hash_features((), self.bins)
        if self.neighbors < 1:
            raise ValueError("hash_knn neighbors는 1 이상이어야 한다")
        if not np.isfinite(self.prior) or self.prior < 0:
            raise ValueError("hash_knn prior는 유한한 0 이상이어야 한다")
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("hash_knn power는 유한한 양수여야 한다")
        if not np.isfinite(self.threshold) or not -1.0 <= self.threshold < 1.0:
            raise ValueError("hash_knn threshold는 -1 이상 1 미만이어야 한다")
        self._base = HashRidgeScore(alpha=self.base_alpha, bins=self.bins)
        self.version = (
            f"hashknn.v1(f={HASH_FEATURE_VERSION},b={self.bins},"
            f"k={self.neighbors},p={self.prior:g},pow={self.power:g},"
            f"t={self.threshold:g},ba={self.base_alpha:g},"
            f"sf={int(self.same_family)})"
        )

    def fit(self, train: Dataset) -> None:
        self._base.fit(train)
        hashed = extract_hash_features(train.texts, self.bins)[:, -self.bins :]
        # Dataset.keys는 prompt 본문의 content hash다. 특징에는 넣지 않고 오직
        # 순서 불변의 저장/tie-break 순서를 만드는 데만 사용한다.
        order = np.argsort(np.asarray(train.keys), kind="stable")
        self._vectors = hashed[order]
        self._scores = np.asarray(train.score, dtype=float)[order]
        self._families = family_codes(train.texts)[order]

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        base = self._base.predict(texts)
        if not texts or len(self._vectors) == 0:
            return base
        query = extract_hash_features(texts, self.bins)[:, -self.bins :]
        query_families = family_codes(texts)
        output = base.copy()
        k = min(self.neighbors, len(self._vectors))

        # 전체 similarity 행렬을 아티팩트에 저장하지 않고 query chunk별로만
        # 만든다. stable argsort는 내용 해시로 정렬된 학습행을 2차 동률 기준으로
        # 사용하므로 실행 순서가 달라도 같은 이웃을 고른다.
        chunk_size = 256
        for start in range(0, len(query), chunk_size):
            stop = min(start + chunk_size, len(query))
            similarity = query[start:stop] @ self._vectors.T
            if self.same_family:
                mismatch = (
                    query_families[start:stop, None] != self._families[None, :]
                )
                similarity[mismatch] = -np.inf
            ranked = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
            selected_similarity = np.take_along_axis(similarity, ranked, axis=1)
            weight = np.maximum(selected_similarity - self.threshold, 0.0)
            weight = np.power(weight, self.power)
            selected_score = self._scores[ranked]
            weight_sum = weight.sum(axis=1)
            denominator = self.prior + weight_sum
            usable = denominator > 0
            if usable.any():
                neighbor_total = np.einsum(
                    "nk,nkm->nm", weight[usable], selected_score[usable]
                )
                rows = np.where(usable)[0] + start
                output[rows] = (
                    self.prior * base[rows] + neighbor_total
                ) / denominator[usable, None]
        return np.clip(output, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "base": self._base.state(),
            "vectors": self._vectors.tolist(),
            "scores": self._scores.tolist(),
            "families": self._families.tolist(),
        }

    def load_state(self, state: dict) -> None:
        self._base.load_state(state["base"])
        self._vectors = np.asarray(state["vectors"], dtype=float)
        self._scores = np.asarray(state["scores"], dtype=float)
        self._families = np.asarray(state["families"], dtype=int)


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


class _BinaryProbabilityModel:
    """계열별 ridge로 이진 사건의 확률을 예측한다.

    분류 정확도 자체가 목적이 아니다. 양성/비양성 조건부 이득을 혼합하기 위한
    확률이므로, 출력은 [0, 1]로 자르고 최종 평가는 라우팅 OOF로 한다.
    """

    def fit(
        self,
        train: Dataset,
        target: np.ndarray,
        alpha: float,
        min_family: int,
    ) -> None:
        codes = family_codes(train.texts)
        features = extract(train.texts)
        target = np.asarray(target, dtype=float)
        self.overall = float(target.mean())
        self.global_fit = _ridge(features, target, alpha)
        self.by_family: Dict[int, tuple] = {}
        self.mean_by_family: Dict[int, float] = {}
        for f in range(len(FAMILIES)):
            mask = codes == f
            if not mask.any():
                continue
            self.mean_by_family[f] = float(target[mask].mean())
            if mask.sum() >= min_family:
                self.by_family[f] = _ridge(features[mask], target[mask], alpha)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        features = extract(texts)
        codes = family_codes(texts)
        out = np.full(len(texts), self.overall)
        for f in np.unique(codes):
            mask = codes == f
            key = int(f)
            if key in self.by_family:
                out[mask] = _ridge_apply(features[mask], self.by_family[key])
            elif key in self.mean_by_family:
                out[mask] = self.mean_by_family[key]
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        def pack(fit):
            mu, sd, weight = fit
            return [mu.tolist(), sd.tolist(), weight.tolist()]

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
        self.mean_by_family = {
            int(k): float(v) for k, v in state["mean_by_family"].items()
        }


@register(SCORE_HEADS, "family_mixture")
class FamilyMixtureScore:
    """``family_useful``의 확률 이중계산을 고친다 (T6).

    ``family_useful``은 문항별 이득을 ``계열평균이득 × P(유용)``으로 본다.
    그런데 계열평균이득에는 **이미 그 계열의 평균 유용확률이 들어 있다.**

        계열평균이득 = P_계열·E[이득|유용] + (1−P_계열)·E[이득|무용]

    거기에 P_문항을 다시 곱하면 확률이 두 번 들어간다. 계열 안에서는
    P_계열이 상수라 순서가 살지만, **배분기는 전 문항을 한 줄로 세우므로
    계열 사이 순서가 왜곡된다.** 유용확률이 낮은 계열이 제곱으로 눌린다.

    올바른 형태는 곱셈이 아니라 조건부 기댓값의 혼합이다.

        ŝ_m = ŝ_light + P_문항·E[이득_m|유용, 계열]
                      + (1−P_문항)·E[이득_m|무용, 계열]

    무용 항을 버리지 않는 이유는 그 값이 0이 아니라 **음수**이기 때문이다.
    어떤 모델도 light를 못 넘는 문항에서 특정 모델은 더 나쁠 수 있다.

    측정 근거: 이득은 거의 이진이다(35.5%만 양성, 양성일 때 0.793±0.251).
    계열은 이득 분산의 11.3%만 설명하고, ``gain>0``의 표본 외 AUC는 0.68이다.
    """

    def __init__(self, alpha: float = 20.0, min_family: int = 40,
                 strength: float = 1.0,
                 active_families: Sequence[str] | None = None) -> None:
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        # 0이면 계열 평균으로 되돌아간다. 1이면 문항별 확률을 그대로 쓴다.
        self.strength = float(strength)
        unknown = set(active_families or ()) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        self.active_families = (
            frozenset(active_families) if active_families is not None else None
        )
        scope = "all" if self.active_families is None else "+".join(
            sorted(self.active_families)
        )
        self.version = (
            f"familymix.v2(a={self.alpha:g},s={self.strength:g},f={scope})"
        )

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        score = np.asarray(train.score, dtype=float)
        useful = (score.max(axis=1) - score[:, 0]) > 0
        gain = score - score[:, [0]]

        n_fam = len(FAMILIES)
        self._light = np.zeros(n_fam)
        self._hit = np.zeros((n_fam, N_MODELS))   # E[이득 | 유용]
        self._miss = np.zeros((n_fam, N_MODELS))  # E[이득 | 무용]
        self._share = np.zeros(n_fam)             # 계열 평균 유용확률

        light_all = float(score[:, 0].mean())
        hit_all = gain[useful].mean(axis=0) if useful.any() else np.zeros(N_MODELS)
        miss_all = gain[~useful].mean(axis=0) if (~useful).any() else np.zeros(N_MODELS)
        for f in range(n_fam):
            m = codes == f
            if not m.any():
                self._light[f] = light_all
                self._hit[f], self._miss[f] = hit_all, miss_all
                self._share[f] = float(useful.mean())
                continue
            self._light[f] = float(score[m, 0].mean())
            self._share[f] = float(useful[m].mean())
            hit, miss = m & useful, m & ~useful
            # 표본이 너무 적은 칸은 전체 평균으로 되돌린다.
            self._hit[f] = gain[hit].mean(axis=0) if hit.sum() >= 5 else hit_all
            self._miss[f] = gain[miss].mean(axis=0) if miss.sum() >= 5 else miss_all

        self._useless = _UselessModel()
        self._useless.fit(train, self.alpha, self.min_family)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        codes = family_codes(texts)
        p_useful = 1.0 - self._useless.predict(texts)
        # strength=0이면 계열 평균 유용확률로 되돌아가 family 헤드와 같아진다.
        strength = np.full(len(texts), self.strength)
        if self.active_families is not None:
            active_codes = {
                i for i, name in enumerate(FAMILIES) if name in self.active_families
            }
            strength = np.array(
                [self.strength if int(code) in active_codes else 0.0 for code in codes]
            )
        p = self._share[codes] + strength * (p_useful - self._share[codes])
        p = np.clip(p, 0.0, 1.0)[:, None]

        gain = p * self._hit[codes] + (1.0 - p) * self._miss[codes]
        out = self._light[codes][:, None] + gain
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "light": self._light.tolist(),
            "hit": self._hit.tolist(),
            "miss": self._miss.tolist(),
            "share": self._share.tolist(),
            "useless": self._useless.state(),
        }

    def load_state(self, state: dict) -> None:
        self._light = np.asarray(state["light"], dtype=float)
        self._hit = np.asarray(state["hit"], dtype=float)
        self._miss = np.asarray(state["miss"], dtype=float)
        self._share = np.asarray(state["share"], dtype=float)
        self._useless = _UselessModel()
        self._useless.load_state(state["useless"])


@register(SCORE_HEADS, "response_shape")
class ResponseShapeScore:
    """두 단계의 계산량 반응을 따로 예측한다.

    단일 ``P(any model beats light)``는 ax31과 K1을 같은 방향으로 움직인다.
    하지만 공개 outcome에는 ``light→ax31``에서 오르는 early-gain과
    ``ax31→K1``에서만 오르는 late-gain이 다르게 나타난다. 두 증분을 분리한다.

    각 단계 k에서 다음 조건부 기댓값을 사용한다.

        E[Δq_k|x] = P(Δq_k>0|x)·E[Δq_k|Δq_k>0,family]
                   + (1-P)·E[Δq_k|Δq_k≤0,family]

    ``strength=0``이면 정확히 계열 평균으로 돌아가고, 선택한 계열에만 문항별
    확률을 적용할 수 있다. 최종 목적은 분류 AUC가 아니라 OOF 라우팅 점수다.
    """

    def __init__(
        self,
        alpha: float = 20.0,
        min_family: int = 40,
        strength: float = 1.0,
        active_families: Sequence[str] | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self.strength = float(strength)
        unknown = set(active_families or ()) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        self.active_families = (
            frozenset(active_families) if active_families is not None else None
        )
        scope = "all" if self.active_families is None else "+".join(
            sorted(self.active_families)
        )
        self.version = (
            f"response.v1(a={self.alpha:g},s={self.strength:g},f={scope})"
        )

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        score = np.asarray(train.score, dtype=float)
        stage_gain = np.column_stack(
            [score[:, 1] - score[:, 0], score[:, 2] - score[:, 1]]
        )
        positive = stage_gain > 0
        n_families = len(FAMILIES)
        self._light = np.zeros(n_families)
        self._share = np.zeros((n_families, 2))
        self._hit = np.zeros((n_families, 2))
        self._miss = np.zeros((n_families, 2))

        light_overall = float(score[:, 0].mean())
        for f in range(n_families):
            family_mask = codes == f
            if not family_mask.any():
                family_mask = np.ones(len(train), dtype=bool)
            self._light[f] = (
                float(score[family_mask, 0].mean()) if family_mask.any()
                else light_overall
            )
            for stage in range(2):
                values = stage_gain[family_mask, stage]
                is_positive = positive[family_mask, stage]
                share = float(is_positive.mean())
                mean = float(values.mean())
                hit = float(values[is_positive].mean()) if is_positive.any() else mean
                miss = float(values[~is_positive].mean()) if (~is_positive).any() else mean
                self._share[f, stage] = share
                self._hit[f, stage] = hit
                self._miss[f, stage] = miss

        self._probability = []
        for stage in range(2):
            model = _BinaryProbabilityModel()
            model.fit(
                train,
                positive[:, stage].astype(float),
                self.alpha,
                self.min_family,
            )
            self._probability.append(model)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        codes = family_codes(texts)
        strength = np.full(len(texts), self.strength)
        if self.active_families is not None:
            active_codes = {
                i for i, name in enumerate(FAMILIES) if name in self.active_families
            }
            strength = np.array(
                [self.strength if int(code) in active_codes else 0.0 for code in codes]
            )

        gains = np.zeros((len(texts), 2))
        for stage, model in enumerate(self._probability):
            predicted = model.predict(texts)
            share = self._share[codes, stage]
            probability = np.clip(share + strength * (predicted - share), 0.0, 1.0)
            gains[:, stage] = (
                probability * self._hit[codes, stage]
                + (1.0 - probability) * self._miss[codes, stage]
            )

        light = self._light[codes]
        out = np.column_stack([light, light + gains[:, 0], light + gains.sum(axis=1)])
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "light": self._light.tolist(),
            "share": self._share.tolist(),
            "hit": self._hit.tolist(),
            "miss": self._miss.tolist(),
            "probability": [model.state() for model in self._probability],
        }

    def load_state(self, state: dict) -> None:
        self._light = np.asarray(state["light"], dtype=float)
        self._share = np.asarray(state["share"], dtype=float)
        self._hit = np.asarray(state["hit"], dtype=float)
        self._miss = np.asarray(state["miss"], dtype=float)
        self._probability = []
        for model_state in state["probability"]:
            model = _BinaryProbabilityModel()
            model.load_state(model_state)
            self._probability.append(model)


_TEMPLATE_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_TEMPLATE_IDENTIFIER = re.compile(r"\b[a-z_][a-z0-9_]*\b")
_TEMPLATE_WORD = re.compile(r"[a-z가-힣]+")
_TEMPLATE_SPACE = re.compile(r"\s+")


def _template_key(text: str, scheme: str) -> str:
    normalized = _TEMPLATE_NUMBER.sub("#", text.lower())
    if scheme == "identifiers":
        normalized = _TEMPLATE_IDENTIFIER.sub("v", normalized)
    elif scheme == "shape":
        normalized = _TEMPLATE_WORD.sub("w", normalized)
    elif scheme != "digits":
        raise ValueError(f"알 수 없는 template 정규화: {scheme!r}")
    normalized = _TEMPLATE_SPACE.sub(" ", normalized).strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


@register(SCORE_HEADS, "template")
class TemplateScore:
    """정규화 템플릿의 공개 Train outcome을 계열 평균에 축소해 사용한다.

    숫자·식별자가 다른 반복 문제를 같은 템플릿으로 묶는다. 아티팩트에는 원문,
    문항 ID나 개별 특징을 넣지 않고 BLAKE2 템플릿 해시와 집계값만 저장한다.
    공개 자료의 정확한 프롬프트/해시 조회는 공식 규칙에서 허용된다.

    ``prior``가 클수록 표본이 적은 템플릿을 계열 평균으로 더 강하게 당긴다.
    미지 템플릿은 언제나 계열 평균으로 폴백한다.
    """

    def __init__(
        self,
        scheme: str = "identifiers",
        prior: float = 2.0,
        strength: float = 1.0,
        active_families: Sequence[str] | None = None,
    ) -> None:
        if scheme not in {"digits", "identifiers", "shape"}:
            raise ValueError(f"알 수 없는 template 정규화: {scheme!r}")
        if prior < 0:
            raise ValueError("template prior는 0 이상이어야 한다")
        self.scheme = scheme
        self.prior = float(prior)
        self.strength = float(strength)
        unknown = set(active_families or ()) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        self.active_families = (
            frozenset(active_families) if active_families is not None else None
        )
        scope = "all" if self.active_families is None else "+".join(
            sorted(self.active_families)
        )
        self.version = (
            f"template.v1(n={self.scheme},p={self.prior:g},"
            f"s={self.strength:g},f={scope})"
        )

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        overall = train.score.mean(axis=0)
        self._family = np.zeros((len(FAMILIES), N_MODELS))
        for f in range(len(FAMILIES)):
            mask = codes == f
            self._family[f] = train.score[mask].mean(axis=0) if mask.any() else overall

        grouped: Dict[str, list] = {}
        for text, score in zip(train.texts, train.score, strict=True):
            key = _template_key(text, self.scheme)
            if key not in grouped:
                grouped[key] = [0, np.zeros(N_MODELS)]
            grouped[key][0] += 1
            grouped[key][1] += score
        self._templates = {
            key: (int(count), total / count)
            for key, (count, total) in grouped.items()
        }

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        codes = family_codes(texts)
        out = self._family[codes].copy()
        for i, (text, code) in enumerate(zip(texts, codes, strict=True)):
            family = FAMILIES[int(code)]
            if self.active_families is not None and family not in self.active_families:
                continue
            found = self._templates.get(_template_key(text, self.scheme))
            if found is None:
                continue
            count, mean = found
            weight = self.strength * count / (count + self.prior)
            weight = min(max(weight, 0.0), 1.0)
            out[i] = self._family[code] + weight * (mean - self._family[code])
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "family": self._family.tolist(),
            "templates": {
                key: [count, mean.tolist()]
                for key, (count, mean) in self._templates.items()
            },
        }

    def load_state(self, state: dict) -> None:
        self._family = np.asarray(state["family"], dtype=float)
        self._templates = {
            key: (int(row[0]), np.asarray(row[1], dtype=float))
            for key, row in state["templates"].items()
        }


@register(SCORE_HEADS, "blend")
class BlendScore:
    """여러 점수 헤드의 정적 가중 평균.

    서로 다른 일반화 오류를 가진 헤드를 작은 비율로 합칠 때만 사용한다.
    각 내부 헤드는 같은 Train fold에서 독립적으로 적합되므로 OOF 경계가
    보존되고, 제출 아티팩트에는 내부 헤드의 집계 상태만 저장된다.
    """

    def __init__(self, heads: Sequence, weights: Sequence[float]) -> None:
        if len(heads) < 2:
            raise ValueError("blend에는 점수 헤드가 둘 이상 필요하다")
        if len(heads) != len(weights):
            raise ValueError("blend heads와 weights 길이가 다르다")
        weight = np.asarray(weights, dtype=float)
        if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError("blend weights는 유한한 0 이상이며 합이 양수여야 한다")
        self._specs = list(heads)
        self._heads = [build_score_head(spec) for spec in self._specs]
        self._weights = weight / weight.sum()
        joined = "+".join(
            f"{w:g}*{head.version}"
            for w, head in zip(self._weights, self._heads, strict=True)
        )
        self.version = f"blend.v1({joined})"

    def fit(self, train: Dataset) -> None:
        for head in self._heads:
            head.fit(train)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        predictions = np.stack([head.predict(texts) for head in self._heads])
        return np.clip(np.tensordot(self._weights, predictions, axes=1), 0.0, 1.0)

    def state(self) -> dict:
        return {"heads": [head.state() for head in self._heads]}

    def load_state(self, state: dict) -> None:
        rows = state["heads"]
        if len(rows) != len(self._heads):
            raise ValueError("blend 아티팩트의 내부 헤드 수가 설정과 다르다")
        for head, row in zip(self._heads, rows, strict=True):
            head.load_state(row)


@register(SCORE_HEADS, "family_blend")
class FamilyBlendScore:
    """선택한 prompt family에서만 보조 점수 헤드를 섞는다.

    전역 blend는 이미 잘 맞는 계열의 순위까지 바꾼다. 이 래퍼는 비활성
    계열에서 base 예측을 그대로 반환하고, Train 진단으로 미리 지정한 계열만
    ``(1-weight) * base + weight * challenger``로 보정한다. family 판별은
    prompt 본문만 사용하며 평가 메타데이터를 보지 않는다.
    """

    def __init__(
        self,
        base,
        challenger,
        weight: float,
        active_families: Sequence[str],
    ) -> None:
        self.weight = float(weight)
        if not np.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValueError("family_blend weight는 0 이상 1 이하여야 한다")
        unknown = set(active_families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 계열: {sorted(unknown)}")
        if not active_families:
            raise ValueError("family_blend active_families는 비어 있으면 안 된다")
        self._active_codes = frozenset(
            i for i, name in enumerate(FAMILIES) if name in active_families
        )
        self._base = build_score_head(base)
        self._challenger = build_score_head(challenger)
        scope = "+".join(sorted(active_families))
        self.version = (
            f"familyblend.v1(w={self.weight:g},f={scope},"
            f"base={self._base.version},challenger={self._challenger.version})"
        )

    def fit(self, train: Dataset) -> None:
        self._base.fit(train)
        self._challenger.fit(train)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        base = self._base.predict(texts)
        active = np.isin(family_codes(texts), tuple(self._active_codes))
        if not active.any() or self.weight == 0.0:
            return base
        challenger = self._challenger.predict(texts)
        out = base.copy()
        out[active] = (
            (1.0 - self.weight) * base[active]
            + self.weight * challenger[active]
        )
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        return {
            "base": self._base.state(),
            "challenger": self._challenger.state(),
        }

    def load_state(self, state: dict) -> None:
        self._base.load_state(state["base"])
        self._challenger.load_state(state["challenger"])


@register(SCORE_HEADS, "tiered")
class TieredScore:
    """budget tier마다 독립적인 점수 헤드를 쓴다.

    tier는 공식 런타임 입력이다. 저예산에서는 작은 이득도 파산 위험보다
    중요하고, Premium에서는 K1 품질 순위가 중요하므로 같은 점수 모델을 모든
    등급에 강제할 이유가 없다. 각 내부 헤드는 같은 Train fold에서 적합된다.
    """

    def __init__(self, heads: Dict[str, object]) -> None:
        missing = set(TIERS) - set(heads)
        extra = set(heads) - set(TIERS)
        if missing or extra:
            raise ValueError(f"tiered heads 등급 오류: 누락={sorted(missing)}, 초과={sorted(extra)}")
        self._specs = {tier: heads[tier] for tier in TIERS}
        self._heads = {
            tier: build_score_head(self._specs[tier]) for tier in TIERS
        }
        joined = ";".join(
            f"{tier}={self._heads[tier].version}" for tier in TIERS
        )
        self.version = f"tieredscore.v1({joined})"

    def fit(self, train: Dataset) -> None:
        for head in self._heads.values():
            head.fit(train)

    def predict_tier(self, texts: Sequence[str], tier: str) -> np.ndarray:
        if tier not in self._heads:
            raise ValueError(f"알 수 없는 tier: {tier!r}")
        return self._heads[tier].predict(texts)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """기존 ScoreHead 계약용 대표값. 실제 할당은 ``predict_tier``를 쓴다."""

        return self.predict_tier(texts, "fast")

    def state(self) -> dict:
        return {"heads": {tier: self._heads[tier].state() for tier in TIERS}}

    def load_state(self, state: dict) -> None:
        rows = state["heads"]
        if set(rows) != set(TIERS):
            raise ValueError("tiered 아티팩트의 등급 목록이 설정과 다르다")
        for tier in TIERS:
            self._heads[tier].load_state(rows[tier])


@register(SCORE_HEADS, "family_useful")
class FamilyUsefulScore:
    """계열 평균 이득을 **문항별 '유용할 확률'로 축소**한다.

    .. warning::

       **확률을 두 번 센다.** ``계열평균이득``에는 이미 그 계열의 평균
       유용확률이 들어 있는데 거기에 문항별 확률을 다시 곱한다. 그 결과
       예측 이득 평균이 0.146에서 0.061로 찌그러지고, 계열 사이 순서가
       왜곡된다. 새로 쓰려면 :class:`FamilyMixtureScore`를 쓴다.
       측정: CV 0.6313 대 ``family`` 0.6349 (2026-08-13).

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


@register(SCORE_HEADS, "ridge")
class RidgeScore:
    """계열별 회귀로 **문항 단위** 점수를 예측한다.

    계열 평균은 같은 계열의 모든 문항에 같은 이득을 준다. 배분기의 ROI 정렬이
    계열 안에서는 비용 차이로만 결정된다는 뜻이다. 측정상 점수 예측 완벽화의
    값어치가 비용 예측의 4배(+0.087 vs +0.022)라 여기가 가장 큰 레버다.

    다만 목표가 noisy하다. ``score``는 2~4회 생성의 평균이라 표본노이즈 sd가
    0.12~0.15이고, 참 기대값에 대한 R²는 0.23~0.31에 그친다. 그래서
    ``shrink``로 계열 평균 쪽으로 당겨 과적합을 막는다.

    ``target='gain'``이면 light 점수와 이득을 따로 예측한다. 이득은 분산이
    작아 목표로 더 안정적일 수 있다 - 어느 쪽이 나은지는 실험으로 정한다.
    """

    def __init__(self, alpha: float = 8.0, min_family: int = 40,
                 shrink: float = 0.0, target: str = "score") -> None:
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self.shrink = float(shrink)
        if target not in {"score", "gain"}:
            raise ValueError(f"target은 'score' 또는 'gain': {target!r}")
        self.target = target
        self.version = (
            f"ridgescore.v1(a={self.alpha:g},sh={self.shrink:g},t={self.target})"
        )

    def fit(self, train: Dataset) -> None:
        codes = family_codes(train.texts)
        features = extract(train.texts)

        if self.target == "gain":
            target = np.column_stack(
                [train.score[:, 0], train.score[:, 1:] - train.score[:, [0]]]
            )
        else:
            target = train.score

        self._table = np.zeros((len(FAMILIES), N_MODELS))
        overall = target.mean(axis=0)
        self._global = [
            _ridge(features, target[:, j], self.alpha) for j in range(N_MODELS)
        ]
        self._by_family: Dict[int, list] = {}
        for f in range(len(FAMILIES)):
            m = codes == f
            self._table[f] = target[m].mean(axis=0) if m.any() else overall
            if m.sum() >= self.min_family:
                self._by_family[f] = [
                    _ridge(features[m], target[m, j], self.alpha)
                    for j in range(N_MODELS)
                ]

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        features = extract(texts)
        codes = family_codes(texts)
        out = self._table[codes].copy()
        for f in np.unique(codes):
            m = codes == f
            fitted = self._by_family.get(int(f))
            if fitted is None:
                continue
            for j in range(N_MODELS):
                predicted = _ridge_apply(features[m], fitted[j])
                # 계열 평균 쪽으로 당겨 노이즈에 과적합하는 것을 막는다.
                out[m, j] = (
                    self.shrink * self._table[f, j] + (1.0 - self.shrink) * predicted
                )
        if self.target == "gain":
            light = np.clip(out[:, [0]], 0.0, 1.0)
            out = np.hstack([light, light + out[:, 1:]])
        return np.clip(out, 0.0, 1.0)

    def state(self) -> dict:
        def pack(fitted):
            return [[m.tolist(), s.tolist(), w.tolist()] for (m, s, w) in fitted]

        return {
            "table": self._table.tolist(),
            "global": pack(self._global),
            "by_family": {str(k): pack(v) for k, v in self._by_family.items()},
        }

    def load_state(self, state: dict) -> None:
        def unpack(rows):
            return [tuple(np.asarray(x, dtype=float) for x in row) for row in rows]

        self._table = np.asarray(state["table"], dtype=float)
        self._global = unpack(state["global"])
        self._by_family = {int(k): unpack(v) for k, v in state["by_family"].items()}


# --------------------------------------------------------------------------
# 비용 헤드
# --------------------------------------------------------------------------


@register(COST_HEADS, "hash_ridge")
class HashRidgeCost:
    """총 monetary cost를 직접 로그 회귀하는 hashed n-gram 비용 헤드.

    출력 토큰을 경유하는 기존 ``ridge``와 달리 공개 채점식으로 계산된 최종
    비용 자체를 목표로 삼는다. ``z``는 필요할 때 승격 비용만 위로 편향하는
    안전 손잡이다. 기본값 0은 공식 베이스라인의 예측과 같다.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        bins: int = 256,
        z: float = 0.0,
        z_light: float = 0.0,
        calibration: bool = False,
        risk_quantile: float | None = None,
        unseen_family_risk: bool = False,
        unseen_risk_boost: float = 1.0,
        risk_oof_folds: int | None = None,
        conditional_risk_families: Sequence[str] = (),
        conditional_risk_alpha: float = 10.0,
        conditional_risk_bins: int = 128,
        conditional_risk_strength: float = 1.0,
        conditional_risk_min_family: int = 40,
    ) -> None:
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.z = float(z)
        self.z_light = float(z_light)
        self.calibration = bool(calibration)
        self.risk_quantile = (
            None if risk_quantile is None else float(risk_quantile)
        )
        self.unseen_family_risk = bool(unseen_family_risk)
        self.unseen_risk_boost = float(unseen_risk_boost)
        self.risk_oof_folds = (
            None if risk_oof_folds is None else int(risk_oof_folds)
        )
        unknown = set(conditional_risk_families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"알 수 없는 conditional risk 계열: {sorted(unknown)}")
        self._conditional_codes = frozenset(
            code
            for code, family in enumerate(FAMILIES)
            if family in conditional_risk_families
        )
        self.conditional_risk_alpha = float(conditional_risk_alpha)
        self.conditional_risk_bins = int(conditional_risk_bins)
        self.conditional_risk_strength = float(conditional_risk_strength)
        self.conditional_risk_min_family = int(conditional_risk_min_family)
        if self.risk_quantile is not None and not 0.5 <= self.risk_quantile < 1.0:
            raise ValueError("risk_quantile은 0.5 이상 1.0 미만이어야 한다")
        if self.risk_oof_folds is not None and self.risk_oof_folds < 2:
            raise ValueError("risk_oof_folds는 2 이상이어야 한다")
        if self.risk_oof_folds is not None and self.risk_quantile is None:
            raise ValueError("risk_oof_folds에는 risk_quantile이 필요하다")
        if self._conditional_codes and self.risk_oof_folds is None:
            raise ValueError("conditional risk에는 risk_oof_folds가 필요하다")
        if self._conditional_codes and self.risk_quantile is None:
            raise ValueError("conditional risk에는 risk_quantile이 필요하다")
        if not np.isfinite(self.conditional_risk_alpha) or self.conditional_risk_alpha <= 0:
            raise ValueError("conditional_risk_alpha는 유한한 양수여야 한다")
        if (
            not np.isfinite(self.conditional_risk_strength)
            or self.conditional_risk_strength < 0
        ):
            raise ValueError("conditional_risk_strength는 유한한 0 이상이어야 한다")
        if self.conditional_risk_min_family < 2:
            raise ValueError("conditional_risk_min_family는 2 이상이어야 한다")
        extract_hash_features((), self.bins)
        extract_hash_features((), self.conditional_risk_bins)
        version = (
            "v6"
            if self._conditional_codes
            else "v5"
            if self.risk_oof_folds is not None
            else ("v4" if self.unseen_family_risk else "v3")
        )
        unseen = ",ur=1" if self.unseen_family_risk else ""
        if self.unseen_risk_boost != 1.0:
            unseen += f",urb={self.unseen_risk_boost:g}"
        risk_oof = (
            f",rof={self.risk_oof_folds}"
            if self.risk_oof_folds is not None
            else ""
        )
        conditional = ""
        if self._conditional_codes:
            scope = "+".join(FAMILIES[code] for code in sorted(self._conditional_codes))
            conditional = (
                f",crf={scope},cra={self.conditional_risk_alpha:g},"
                f"crb={self.conditional_risk_bins},"
                f"crs={self.conditional_risk_strength:g},"
                f"crm={self.conditional_risk_min_family}"
            )
        self.version = (
            f"hashcost.{version}(f={HASH_FEATURE_VERSION},a={self.alpha:g},"
            f"b={self.bins},z={self.z:g},zl={self.z_light:g},"
            f"cal={int(self.calibration)},rq={self.risk_quantile}{unseen}"
            f"{risk_oof}{conditional})"
        )

    # 통합 로그 비용 회귀 기준을 갈아끼울 수 있는 봉합(seam). 기본은 공식
    # 베이스라인과 같은 표준화 ridge이고, GBM 파생 클래스만 셋을 재정의한다.
    # 계약: fit_logcost(design, log_cost) -> fitted ;
    #        apply_logcost(design, fitted) -> log_cost_pred ;  fitted는 opaque.
    def _fit_logcost(self, design, log_cost):
        return _hash_ridge_fit(design, log_cost, self.alpha)

    def _apply_logcost(self, design, fitted):
        return _hash_ridge_apply(design, fitted)

    def _oof_logcost(self, design, log_cost, folds):
        return _hash_ridge_oof_apply(design, log_cost, folds, self.alpha)

    def fit(self, train: Dataset) -> None:
        design = extract_hash_features(train.texts, self.bins)
        log_cost = np.log(train.cost)
        self._fitted = self._fit_logcost(design, log_cost)
        residual = log_cost - self._apply_logcost(design, self._fitted)
        self._sd_log = residual.std(axis=0)
        # 비공개 입력의 이상치 하나가 exp를 폭주시켜 전체를 all-light로 만드는
        # 것을 막되, 학습 범위 주변의 정상 외삽은 허용한다.
        margin = np.log(2.0)
        self._log_lo = log_cost.min(axis=0) - margin
        self._log_hi = log_cost.max(axis=0) + margin

        # 로그 평균을 exp로 되돌리면 heavy tail의 산술 평균을 체계적으로
        # 과소추정한다(Jensen). 예산은 개별 중앙값이 아니라 **총합**에 걸리므로
        # 프롬프트 계열·모델별 학습 총합이 맞도록 배율을 보정한다.
        raw_cost = np.exp(self._apply_logcost(design, self._fitted))
        codes = family_codes(train.texts)
        self._seen_families = np.bincount(
            codes, minlength=len(FAMILIES)
        ).astype(bool)
        global_calibration = train.cost.sum(axis=0) / np.maximum(
            raw_cost.sum(axis=0), 1e-12
        )
        self._calibration = np.tile(global_calibration, (len(FAMILIES), 1))
        for family in range(len(FAMILIES)):
            mask = codes == family
            if mask.sum() >= 40:
                self._calibration[family] = train.cost[mask].sum(axis=0) / np.maximum(
                    raw_cost[mask].sum(axis=0), 1e-12
                )
        self._calibration = np.clip(self._calibration, 0.2, 5.0)

        # 평균 보정으로도 생성 폭주 꼬리는 남는다. family/model 조합별
        # 실제/예측 배율의 상위 분위로 그 조합의 empirical tail risk를 가격에
        # 넣는다. 문항별 outcome lookup이 아니라 Train에서 학습한 정적 통계다.
        risk_cost = raw_cost
        if self.risk_oof_folds is not None:
            oof_log_cost = self._oof_logcost(
                design,
                log_cost,
                train.folds(self.risk_oof_folds),
            )
            risk_cost = np.exp(np.clip(oof_log_cost, self._log_lo, self._log_hi))
        actual_relative = train.cost / np.maximum(train.cost[:, [0]], 1e-12)
        predicted_relative = risk_cost / np.maximum(risk_cost[:, [0]], 1e-12)
        ratio = actual_relative / np.maximum(predicted_relative, 1e-12)
        quantile = self.risk_quantile if self.risk_quantile is not None else 0.5
        global_risk = np.quantile(ratio, quantile, axis=0)
        self._risk = np.tile(global_risk, (len(FAMILIES), 1))
        for family in range(len(FAMILIES)):
            mask = codes == family
            if mask.sum() >= 40:
                self._risk[family] = np.quantile(ratio[mask], quantile, axis=0)
        # 위에서 이미 문항별 upgrade/light 상대오차를 만들었다. 위험 pricing은
        # 안전 방향으로만 작동하도록 승격 배율의 하한을 1로 둔다.
        self._risk[:, 0] = 1.0
        self._risk[:, 1:] = np.clip(self._risk[:, 1:], 1.0, 20.0)
        self._unseen_risk = self._risk.max(axis=0) * self.unseen_risk_boost
        self._unseen_risk[0] = 1.0  # light는 부스트하지 않는다 (비율 분모 보존)

        # family q분위 하나는 같은 계열의 모든 문항을 똑같이 비싸게 만든다.
        # q분위까지는 위 scalar risk가 이미 가격에 넣었으므로, 남은 양의
        # log 초과분만 prompt lexical 특징으로 예측한다. family 평균을 빼고
        # 양수만 쓰면 평균적인 문항은 기존 비용과 정확히 같고, 설명 가능한
        # tail만 추가로 비싸진다.
        self._conditional_by_family: Dict[int, tuple] = {}
        self._conditional_center: Dict[int, np.ndarray] = {}
        if self._conditional_codes:
            conditional_design = extract_hash_features(
                train.texts, self.conditional_risk_bins
            )
            excess = np.log(
                np.maximum(
                    ratio / np.maximum(self._risk[codes], 1e-12),
                    1.0,
                )
            )
            excess[:, 0] = 0.0
            for code in self._conditional_codes:
                mask = codes == code
                if mask.sum() < self.conditional_risk_min_family:
                    continue
                self._conditional_by_family[code] = _hash_ridge_fit(
                    conditional_design[mask],
                    excess[mask],
                    self.conditional_risk_alpha,
                )
                self._conditional_center[code] = excess[mask].mean(axis=0)

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        design = extract_hash_features(texts, self.bins)
        log_cost = self._apply_logcost(design, self._fitted)
        bias = np.full(N_MODELS, self.z)
        bias[0] = self.z_light
        log_cost = np.clip(
            log_cost + bias * self._sd_log,
            self._log_lo,
            self._log_hi,
        )
        cost = np.exp(log_cost)
        if self.calibration:
            cost = cost * self._calibration[family_codes(texts)]
        if self.risk_quantile is not None:
            codes = family_codes(texts)
            risk = self._risk[codes].copy()
            if self.unseen_family_risk:
                risk[~self._seen_families[codes]] = self._unseen_risk
            cost = cost * risk
        if self._conditional_by_family and self.conditional_risk_strength:
            conditional_design = extract_hash_features(
                texts, self.conditional_risk_bins
            )
            codes = family_codes(texts)
            correction = np.ones_like(cost)
            for code, fitted in self._conditional_by_family.items():
                mask = codes == code
                if not mask.any():
                    continue
                log_excess = _hash_ridge_apply(conditional_design[mask], fitted)
                log_excess = np.maximum(
                    log_excess - self._conditional_center[code],
                    0.0,
                )
                log_excess[:, 0] = 0.0
                correction[mask] = np.exp(
                    np.clip(
                        self.conditional_risk_strength * log_excess,
                        0.0,
                        np.log(20.0),
                    )
                )
            cost = cost * correction
        # 후보 모델은 정책상 싼 순서다. 회귀의 작은 교차를 그대로 두면
        # 포락선에서 음수 추가비용이 생기므로 단조성을 강제한다.
        cost[:, 1] = np.maximum(cost[:, 1], cost[:, 0] * (1.0 + 1e-12))
        cost[:, 2] = np.maximum(cost[:, 2], cost[:, 1] * (1.0 + 1e-12))
        sd = cost * np.expm1(self._sd_log)[None, :]
        return cost, sd

    def state(self) -> dict:
        return {
            "fitted": _pack_hash_ridge(self._fitted),
            "sd_log": self._sd_log.tolist(),
            "log_lo": self._log_lo.tolist(),
            "log_hi": self._log_hi.tolist(),
            "calibration": self._calibration.tolist(),
            "risk": self._risk.tolist(),
            "conditional_by_family": {
                str(code): _pack_hash_ridge(fitted)
                for code, fitted in self._conditional_by_family.items()
            },
            "conditional_center": {
                str(code): center.tolist()
                for code, center in self._conditional_center.items()
            },
            **(
                {
                    "seen_families": self._seen_families.tolist(),
                    "unseen_risk": self._unseen_risk.tolist(),
                }
                if self.unseen_family_risk
                else {}
            ),
        }

    def load_state(self, state: dict, policy) -> None:
        self._fitted = _unpack_hash_ridge(state["fitted"])
        self._sd_log = np.asarray(state["sd_log"], dtype=float)
        self._log_lo = np.asarray(state["log_lo"], dtype=float)
        self._log_hi = np.asarray(state["log_hi"], dtype=float)
        self._calibration = np.asarray(state["calibration"], dtype=float)
        self._risk = np.asarray(state["risk"], dtype=float)
        self._conditional_by_family = {
            int(code): _unpack_hash_ridge(fitted)
            for code, fitted in state.get("conditional_by_family", {}).items()
        }
        self._conditional_center = {
            int(code): np.asarray(center, dtype=float)
            for code, center in state.get("conditional_center", {}).items()
        }
        if self.unseen_family_risk:
            self._seen_families = np.asarray(state["seen_families"], dtype=bool)
            self._unseen_risk = np.asarray(state["unseen_risk"], dtype=float)


@register(COST_HEADS, "gbm_hash_ridge")
class GBMHashRidgeCost(HashRidgeCost):
    """HashRidgeCost와 동일한 risk/calibration/unseen 머신을 쓰되, 로그 비용
    회귀의 기준을 Gradient Boosting(HistGBM, 학습 전용)으로 대체한 실험 헤드.

    이 설계는 경쟁사가 독립적으로 검증한 '릿지+비선형 비용 예측을 섞으면
    예산이 건강해진다'는 방향을 우리 매커니즘에 적용한 것(아이디어만 참조).
    sklearn은 **학습 단계에서만** 쓰고, 추론은 pickle로 저장된 estimator를
    순수 파이썬으로 옮기지 않는 한 런타임에 sklearn이 필요하다. 따라서 이
    헤드는 'GBM이 실제로 dev 점수를 올리는가'를 판정하는 실험 헤드로만 쓰고,
    채택 시 순수 파이썬 트리 inference로 전환한다.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        bins: int = 256,
        z: float = 0.0,
        z_light: float = 0.0,
        calibration: bool = False,
        risk_quantile: float | None = None,
        unseen_family_risk: bool = False,
        unseen_risk_boost: float = 1.0,
        risk_oof_folds: int | None = None,
        conditional_risk_families: Sequence[str] = (),
        conditional_risk_alpha: float = 10.0,
        conditional_risk_bins: int = 128,
        conditional_risk_strength: float = 1.0,
        conditional_risk_min_family: int = 40,
        # GBM 전용
        gbm_max_leaf_nodes: int = 15,
        gbm_min_samples_leaf: int = 10,
        gbm_max_iter: int = 100,
        gbm_learning_rate: float = 0.1,
        gbm_l2: float = 1.0,
        # 0<=gbm_w<=1: >0이면 ridge와 GBM의 로그비용 예측을 평균한다
        # (gbm_w=0 → 순수 ridge, gbm_w=1 → 순수 GBM).
        gbm_ensemble_w: float = 1.0,
    ) -> None:
        super().__init__(
            alpha=alpha, bins=bins, z=z, z_light=z_light,
            calibration=calibration, risk_quantile=risk_quantile,
            unseen_family_risk=unseen_family_risk,
            unseen_risk_boost=unseen_risk_boost,
            risk_oof_folds=risk_oof_folds,
            conditional_risk_families=conditional_risk_families,
            conditional_risk_alpha=conditional_risk_alpha,
            conditional_risk_bins=conditional_risk_bins,
            conditional_risk_strength=conditional_risk_strength,
            conditional_risk_min_family=conditional_risk_min_family,
        )
        self.gbm_max_leaf_nodes = int(gbm_max_leaf_nodes)
        self.gbm_min_samples_leaf = int(gbm_min_samples_leaf)
        self.gbm_max_iter = int(gbm_max_iter)
        self.gbm_learning_rate = float(gbm_learning_rate)
        self.gbm_l2 = float(gbm_l2)
        self.gbm_ensemble_w = float(gbm_ensemble_w)
        if not 0.0 <= self.gbm_ensemble_w <= 1.0:
            raise ValueError("gbm_ensemble_w는 0..1 이어야 한다")
        self.version = "gbmhashcost.probe(urb=%g,gmm=%d,gml=%d,gmi=%d,glr=%g,gl2=%g,w=%g)" % (
            self.unseen_risk_boost, self.gbm_max_leaf_nodes,
            self.gbm_min_samples_leaf, self.gbm_max_iter,
            self.gbm_learning_rate, self.gbm_l2, self.gbm_ensemble_w,
        )

    def _gbm_estimator(self, seed: int):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            l2_regularization=self.gbm_l2,
            max_leaf_nodes=self.gbm_max_leaf_nodes,
            min_samples_leaf=self.gbm_min_samples_leaf,
            max_iter=self.gbm_max_iter,
            learning_rate=self.gbm_learning_rate,
            random_state=seed,
        )

    def _fit_logcost(self, design, log_cost):
        estimators = []
        ridge_fitted = None
        if 0.0 <= self.gbm_ensemble_w < 1.0:
            ridge_fitted = _hash_ridge_fit(design, log_cost, self.alpha)
        for m in range(log_cost.shape[1]):
            g = self._gbm_estimator(0 + m)
            g.fit(design, log_cost[:, m])
            estimators.append(g)
        if self.gbm_ensemble_w >= 1.0:
            return estimators, None
        return estimators, ridge_fitted

    def _apply_logcost(self, design, fitted):
        estimators, ridge_fitted = fitted
        gbm_pred = np.column_stack([g.predict(design) for g in estimators])
        if ridge_fitted is None:
            return gbm_pred
        ridge_pred = _hash_ridge_apply(design, ridge_fitted)
        return self.gbm_ensemble_w * gbm_pred + (1.0 - self.gbm_ensemble_w) * ridge_pred

    def _oof_logcost(self, design, log_cost, folds):
        pred = np.full_like(log_cost, np.nan, dtype=float)
        all_rows = np.arange(len(design))
        coverage = np.zeros(len(design), dtype=int)
        for held_out in folds:
            held_out = np.asarray(held_out, dtype=int)
            if len(held_out) == 0:
                continue
            tr = all_rows[~np.isin(all_rows, held_out)]
            # OOF에서는 ensemble의 ridge 부분도 train fold에서 다시 적합해야
            # 하므로, 함수형 경로를 그대로 쓴다.
            fitted = self._fit_logcost(design[tr], log_cost[tr])
            pred[held_out] = self._apply_logcost(design[held_out], fitted)
            coverage[held_out] += 1
        if not np.all(coverage == 1):
            raise ValueError("OOF GBM fold가 모든 학습 행을 정확히 포함해야 한다")
        return pred

    def state(self) -> dict:
        raise NotImplementedError(
            "GBMHashRidgeCost는 실험 전용(probe) 헤드다. 채택 시 순수 파이썬 "
            "트리 inference로 전환한 뒤 직렬화한다."
        )

    def load_state(self, state: dict, policy) -> None:
        raise NotImplementedError(
            "GBMHashRidgeCost는 실험 전용(probe) 헤드다."
        )


@register(COST_HEADS, "tiered")
class TieredCost:
    """공식 budget tier마다 독립적인 비용 위험 가격을 사용한다.

    실제 monetary cost 자체는 tier와 무관하지만, 초과 손실과 쓸 수 있는
    headroom은 tier마다 다르다. Fast의 희소 tail을 비싸게 보는 보정이
    Balanced/Premium의 정상 ROI 순위까지 바꾸지 않도록 적합 상태를 분리한다.
    각 내부 헤드는 prompt만 보고, tier는 공식 런타임 인자로만 분기한다.
    """

    def __init__(self, heads: Dict[str, object]) -> None:
        missing = set(TIERS) - set(heads)
        extra = set(heads) - set(TIERS)
        if missing or extra:
            raise ValueError(
                f"tiered cost heads 등급 오류: 누락={sorted(missing)}, "
                f"초과={sorted(extra)}"
            )
        self._specs = {tier: heads[tier] for tier in TIERS}
        self._heads = {
            tier: build_cost_head(self._specs[tier]) for tier in TIERS
        }
        joined = ";".join(
            f"{tier}={self._heads[tier].version}" for tier in TIERS
        )
        self.version = f"tieredcost.v1({joined})"

    def fit(self, train: Dataset) -> None:
        for head in self._heads.values():
            head.fit(train)

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        return self.predict_tier(texts, "fast")

    def predict_tier(
        self, texts: Sequence[str], tier: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if tier not in self._heads:
            raise ValueError(f"알 수 없는 tier: {tier!r}")
        return self._heads[tier].predict(texts)

    def state(self) -> dict:
        return {"heads": {tier: self._heads[tier].state() for tier in TIERS}}

    def load_state(self, state: dict, policy) -> None:
        rows = state["heads"]
        if set(rows) != set(TIERS):
            raise ValueError("tiered cost 아티팩트의 tier 구성이 다르다")
        for tier in TIERS:
            self._heads[tier].load_state(rows[tier], policy)


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
        pool_size: int = 128,
    ) -> None:
        self.pool_size = int(pool_size)
        self.z = float(z)
        self.z_light = float(z_light)
        self.alpha = float(alpha)
        self.min_family = int(min_family)
        self.smearing = bool(smearing)
        self.version = (
            f"ridgecost.v4(z={self.z:g},zl={self.z_light:g},"
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
        take = np.linspace(0, len(global_residual) - 1,
                           min(len(global_residual), self.pool_size))
        self._global_pool = np.sort(global_residual, axis=0)[np.round(take).astype(int)]

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
        self._pool_by_family: Dict[int, np.ndarray] = {}
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
            # 경험 잔차 풀. CLT가 heavy tail을 못 담으므로 실제 분포를
            # 그대로 재표집한다. 크기를 제한해 산출물이 커지지 않게 한다.
            take = np.linspace(0, len(residual) - 1, min(len(residual), self.pool_size))
            self._pool_by_family[f] = np.sort(residual, axis=0)[np.round(take).astype(int)]
        self._policy = train.policy

        # 총합 보정은 계열 회귀가 다 준비된 뒤에 계산한다.
        # smearing은 개별 문항의 Jensen 편향을 고치지만 mean(exp(잔차))가
        # 꼬리에 지배돼 총합을 과대추정한다(실측 1.127). 예산은 총합으로
        # 걸리므로 학습 총합이 맞도록 계열·모델별 배율을 한 번 더 곱한다.
        self._calibration = np.ones((len(FAMILIES), N_MODELS))
        base = self._base_tokens(train.texts)
        global_calib = train.output_tokens.sum(axis=0) / np.maximum(base.sum(axis=0), 1e-9)
        for f in range(len(FAMILIES)):
            m = codes == f
            self._calibration[f] = (
                train.output_tokens[m].sum(axis=0) / np.maximum(base[m].sum(axis=0), 1e-9)
                if m.sum() >= self.min_family
                else global_calib
            )
        self._calibration = np.clip(self._calibration, 0.2, 5.0)

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

        # 회귀 출력을 학습에서 본 범위로 먼저 묶는다.
        log_out = np.clip(log_out, self._log_lo, self._log_hi)

        # 1) 기저 예측. smearing은 exp(잔차)의 평균이라 되돌리기 전에 곱한다.
        base = np.clip(np.exp(log_out) * smear - 1.0, 0.0, self._token_hi)

        # 2) 총합 보정. smearing은 개별 문항의 Jensen 편향은 고치지만
        #    mean(exp(잔차))가 꼬리에 지배돼 **총합을 과대추정**한다.
        #    실측: 끄면 0.870, 켜면 1.127. 예산은 총합으로 걸리므로
        #    학습에서 총합이 맞도록 계열·모델별 배율을 한 번 더 곱한다.
        base = base * self._calibration[codes]

        # 3) 안전 편향. 여기서부터는 의도적으로 위로 틀린다 (RULES C2).
        bias = np.full(N_MODELS, self.z)
        bias[0] = self.z_light
        tok_out = np.clip(base * np.exp(bias * sd_log), 0.0, self._token_hi)
        cost = cost_from_tokens(tok_in, tok_out, self._policy)
        # 산포는 로그 공간 잔차를 비용 단위로 옮겨 근사한다.
        spread_tokens = np.clip(
            np.expm1(log_out + sd_log) - np.expm1(log_out), 0.0, self._token_hi
        )
        sd = cost_from_tokens(np.zeros_like(tok_in), spread_tokens, self._policy)
        return cost, sd

    def _base_tokens(self, texts: Sequence[str]) -> np.ndarray:
        """보정·편향을 얹기 전의 출력 토큰 예측."""

        features = extract(texts)
        codes = family_codes(texts)
        log_out = np.zeros((len(texts), N_MODELS))
        smear = np.ones((len(texts), N_MODELS))
        for f in np.unique(codes):
            m = codes == f
            fitted = self._by_family.get(int(f), self._global)
            for j in range(N_MODELS):
                log_out[m, j] = _ridge_apply(features[m], fitted[j])
            if self.smearing:
                smear[m] = self._smear_by_family.get(int(f), self._global_smear)
        log_out = np.clip(log_out, self._log_lo, self._log_hi)
        return np.clip(np.exp(log_out) * smear - 1.0, 0.0, self._token_hi)

    def residual_multipliers(
        self, texts: Sequence[str], keys: Sequence[str], draws: int
    ) -> np.ndarray:
        """[n, 3, draws] 곱셈 잔차. 난수를 쓰지 않는다.

        문항마다 콘텐츠 해시로 잔차 풀의 시작 위치를 정해 회전시킨다.
        같은 프롬프트는 항상 같은 표본을 받고, 입력 순서와 무관하다 (RULES B).
        """

        codes = family_codes(texts)
        out = np.empty((len(texts), N_MODELS, draws), dtype=float)
        offsets = np.array([int(k[:8], 16) for k in keys], dtype=np.int64)
        for f in np.unique(codes):
            m = codes == f
            pool = self._pool_by_family.get(int(f), self._global_pool)
            size = len(pool)
            take = (offsets[m][:, None] + np.arange(draws)[None, :]) % size
            for j in range(N_MODELS):
                out[m, j, :] = np.exp(pool[:, j][take])
        # smearing이 평균을 이미 보정했으므로 배율의 평균을 1로 맞춘다.
        return out / np.maximum(out.mean(axis=2, keepdims=True), 1e-12)

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
            "global_pool": self._global_pool.tolist(),
            "pool_by_family": {str(f): v.tolist() for f, v in self._pool_by_family.items()},
            "calibration": self._calibration.tolist(),
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
        self._global_pool = np.asarray(state["global_pool"], dtype=float)
        self._pool_by_family = {
            int(k): np.asarray(v, dtype=float) for k, v in state["pool_by_family"].items()
        }
        self._calibration = np.asarray(state["calibration"], dtype=float)
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


@register(GATES, "tail_exposure")
class TailExposureGate:
    """반복될 때 등급 전체를 무너뜨리는 희소 prompt tail을 차단한다.

    총액 예산은 평균적으로 안전해도, 같은 프롬프트가 재표집되면 단일 비용
    예측 실패가 여러 번 복제된다. Train/Dev 실패 분석에서 ax31은 초대형 수치
    프롬프트, K1은 깊게 중첩된 LaTeX 수식에서 이 현상이 집중됐다. 공식 입력인
    tier에 따라 실제로 위험한 모델만 막고 나머지 승격은 보존한다.
    """

    def __init__(
        self,
        max_log_number: float = 40.0,
        digit_ratio: float = 0.6,
        min_latex: float = 1.0,
        min_paren_depth: float = 3.0,
        fast_code_ax31_cap: float = float("inf"),
        balanced_code_ax31_cap: float = float("inf"),
        premium_code_k1_cap: float = float("inf"),
        block_code_k1: bool = False,
        block_other_k1: bool = False,
        k1_item_cap: float | None = None,
        inner: str = "none",
        **inner_kwargs,
    ) -> None:
        self.max_log_number = float(max_log_number)
        self.digit_ratio = float(digit_ratio)
        self.min_latex = float(min_latex)
        self.min_paren_depth = float(min_paren_depth)
        self.fast_code_ax31_cap = float(fast_code_ax31_cap)
        self.balanced_code_ax31_cap = float(balanced_code_ax31_cap)
        self.premium_code_k1_cap = float(premium_code_k1_cap)
        self.block_code_k1 = bool(block_code_k1)
        self.block_other_k1 = bool(block_other_k1)
        self.k1_item_cap = None if k1_item_cap is None else float(k1_item_cap)
        self._inner = GATES[inner](**inner_kwargs)
        self.version = (
            "tailexposure.v1("
            f"num={self.max_log_number:g},digit={self.digit_ratio:g},"
            f"latex={self.min_latex:g},depth={self.min_paren_depth:g},"
            f"code31={self.fast_code_ax31_cap:g},"
            f"balcode31={self.balanced_code_ax31_cap:g},"
            f"codek1={self.premium_code_k1_cap:g},"
            f"blockcodek1={int(self.block_code_k1)},"
            f"blockotherk1={int(self.block_other_k1)})+"
            f"{self._inner.version}"
        )

    def fit(self, train: Dataset) -> None:
        self._inner.fit(train)

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        """기존 Gate 계약용 대표값. 실제 경로는 ``allow_tier``를 쓴다."""

        return self.allow_tier(texts, s_hat, c_hat, "fast")

    def allow_tier(self, texts, s_hat, c_hat, tier: str) -> np.ndarray:
        if tier not in TIERS:
            raise ValueError(f"알 수 없는 tier: {tier!r}")
        allow = self._inner.allow(texts, s_hat, c_hat).copy()
        features = extract(texts)
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        numeric_tail = (
            features[:, index["log_max_number"]] >= self.max_log_number
        ) | (features[:, index["digit_ratio"]] >= self.digit_ratio)
        deep_latex = (
            features[:, index["latex_count"]] >= self.min_latex
        ) & (features[:, index["paren_depth"]] >= self.min_paren_depth)
        codes = family_codes(texts)
        code = codes == FAMILIES.index("code_io")
        other = codes == FAMILIES.index("other")
        relative = c_hat / np.maximum(c_hat[:, [0]], 1e-12)
        if self.block_code_k1:
            allow[code, 2] = False
        if self.block_other_k1:
            allow[other, 2] = False
        if tier in ("fast", "balanced"):
            allow[numeric_tail, 1] = False
        if tier == "fast":
            allow[code & (relative[:, 1] > self.fast_code_ax31_cap), 1] = False
        if tier == "balanced":
            allow[
                code & (relative[:, 1] > self.balanced_code_ax31_cap), 1
            ] = False
        if tier == "premium":
            allow[deep_latex, 2] = False
            allow[code & (relative[:, 2] > self.premium_code_k1_cap), 2] = False
        if self.k1_item_cap is not None:
            # 단일 K1 문항의 예측 비용이 배치 총 light 비용의 k1_item_cap
            # 배를 넘으면 K1을 차단한다. 예측 오차와 무관하게 한 문항이 예산을
            # 독점하는 것을 막는 leelang7 식 손잡이다.
            light_total = float(c_hat[:, 0].sum())
            item_limit = light_total * self.k1_item_cap
            allow[c_hat[:, 2] > item_limit, 2] = False
        allow[:, 0] = True
        return allow

    def state(self) -> dict:
        return {"inner": self._inner.state()}

    def load_state(self, state: dict) -> None:
        self._inner.load_state(state["inner"])


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
        self.version = f"k1cap.v2(p={self.percentile:g},roi={self.min_roi:g})"
        self._roi_gate = FamilyRoiGate(min_roi=min_roi)

    def fit(self, train: Dataset) -> None:
        self._roi_gate.fit(train)

    def allow(self, texts, s_hat, c_hat) -> np.ndarray:
        allow = self._roi_gate.allow(texts, s_hat, c_hat)
        # 임계값은 **예측 분포**에서 잡는다. 학습의 실제 비율로 잡으면
        # 단위가 어긋난다 - z 편향이 K1/light 예측 비율을 8배 이상 부풀리므로
        # 실제 분위와 비교하면 거의 전부 차단된다. 실측: code_io 119문항이
        # 예측 이득 +0.40을 갖고도 통째로 막혀 있었다.
        ratio = c_hat[:, 2] / np.maximum(c_hat[:, 0], 1e-12)
        cap = float(np.percentile(ratio, self.percentile))
        allow[:, 2] &= ratio <= cap
        allow[:, 0] = True
        return allow

    def state(self) -> dict:
        return {"percentile": self.percentile, "roi": self._roi_gate.state()}

    def load_state(self, state: dict) -> None:
        self.percentile = float(state["percentile"])
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
