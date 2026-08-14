# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""프롬프트 내용만 쓰는 signed feature-hashing 특징.

공식 ``hash_regex`` 베이스라인의 강한 부분은 선택 규칙이 아니라, 작은 정규식
특징과 word unigram/bigram을 고정 해시 공간에 넣는 표현이다. 이 모듈은 그
공개 특징 스키마를 참가자 파이프라인의 ``Sequence[str]`` 계약으로 옮긴다.

문항 ID, 데이터 출처, 입력 순서, 후보 모델 출력은 보지도 받지도 않는다.
FNV-1a를 직접 쓰므로 Python 프로세스의 ``hash()`` seed에도 영향받지 않는다.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Sequence

import numpy as np

FEATURE_VERSION = "hash-regex-text.v1"
DEFAULT_HASH_BINS = 256
MIN_HASH_BINS = 16
MAX_HASH_BINS = 16_384

_FNV_OFFSET = 14_695_981_039_346_656_037
_FNV_PRIME = 1_099_511_628_211
_UINT64_MASK = (1 << 64) - 1

_TOKEN = re.compile(r"[A-Za-z]+|[가-힣]+|\d+|[^\w\s]", re.UNICODE)
_WORD = re.compile(r"[A-Za-z가-힣]+")
_SENTENCE_END = re.compile(r"[.!?。！？]")
_NUMBER = re.compile(r"\d")
_CODE_MARKERS = re.compile(
    r"```|(?:^|\s)(?:def|class|function|SELECT|FROM|import|#include)\b|"
    r"[{};]\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MATH_MARKERS = re.compile(r"[=+\-*/^∑∫√≈≠≤≥<>]|\\(?:frac|sum|int|sqrt)\b")
_REASONING_WORDS = re.compile(
    r"\b(?:prove|derive|reason|analyze|explain why|algorithm|complexity|"
    r"증명|유도|추론|분석|알고리즘|복잡도)\b",
    re.IGNORECASE,
)
_FORMAL_REASONING = re.compile(
    r"\b(?:prove|derive|theorem|lemma|counterexample|induction|"
    r"증명|유도|정리|보조정리|반례|귀납)\b",
    re.IGNORECASE,
)
_PROGRAM_ANALYSIS = re.compile(
    r"```|\b(?:traceback|exception|complexity|big[- ]?o|"
    r"시간\s*복잡도|공간\s*복잡도|예외|스택\s*추적)\b",
    re.IGNORECASE,
)
_MULTI_CONSTRAINT = re.compile(
    r"\b(?:exactly|at least|at most|must|only|without|"
    r"정확히|이상|이하|반드시|오직|제외하고)\b",
    re.IGNORECASE,
)
_SIMPLE_TRANSFORM = re.compile(
    r"\b(?:summari[sz]e|rewrite|translate|list|extract|"
    r"요약|바꾸|번역|나열|추출)\b",
    re.IGNORECASE,
)

DENSE_FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_message_count",
    "hangul_ratio",
    "log_code_marker_count",
    "log_math_marker_count",
    "numeric_density",
    "long_context",
    "log_reasoning_marker_count",
    "formal_reasoning",
    "program_analysis",
    "log_multi_constraint_count",
    "simple_transform",
)


def _validate_bins(hash_bins: int) -> None:
    if (
        isinstance(hash_bins, bool)
        or not isinstance(hash_bins, int)
        or not MIN_HASH_BINS <= hash_bins <= MAX_HASH_BINS
        or hash_bins & (hash_bins - 1)
    ):
        raise ValueError("hash_bins는 허용 범위의 2의 거듭제곱이어야 한다")


def _stable_hash(value: str) -> int:
    digest = _FNV_OFFSET
    for byte in value.encode("utf-8"):
        digest ^= byte
        digest = (digest * _FNV_PRIME) & _UINT64_MASK
    return digest


def _normalized_tokens(text: str) -> tuple[str, ...]:
    result = []
    for token in _TOKEN.findall(text):
        normalized = token.casefold()
        if normalized.isdecimal():
            normalized = "<number>"
        result.append(normalized)
    return tuple(result)


@lru_cache(maxsize=4096)
def _raw_feature_vector(text: str, hash_bins: int) -> tuple[float, ...]:
    _validate_bins(hash_bins)
    characters = len(text)
    nonspace = sum(not character.isspace() for character in text)
    hangul = sum("\uac00" <= character <= "\ud7a3" for character in text)
    dense = (
        math.log1p(characters),
        math.log1p(len(_WORD.findall(text))),
        math.log1p(max(1, len(_SENTENCE_END.findall(text)))),
        math.log1p(1),  # 공개 Train/Dev는 모두 단일 prompt/message다.
        hangul / max(1, nonspace),
        math.log1p(len(_CODE_MARKERS.findall(text))),
        math.log1p(len(_MATH_MARKERS.findall(text))),
        len(_NUMBER.findall(text)) / max(1, nonspace),
        float(characters >= 8_000),
        math.log1p(len(_REASONING_WORDS.findall(text))),
        float(bool(_FORMAL_REASONING.search(text))),
        float(bool(_PROGRAM_ANALYSIS.search(text))),
        math.log1p(len(_MULTI_CONSTRAINT.findall(text))),
        float(bool(_SIMPLE_TRANSFORM.search(text))),
    )

    bins = [0.0] * hash_bins
    tokens = _normalized_tokens(text)
    hashed = [f"w1:{token}" for token in tokens]
    hashed.extend(
        f"w2:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:])
    )
    for value in hashed:
        digest = _stable_hash(value)
        index = digest & (hash_bins - 1)
        bins[index] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return dense + tuple(bins)


def extract_hash_features(
    texts: Sequence[str], hash_bins: int = DEFAULT_HASH_BINS
) -> np.ndarray:
    """``[n, 14 + hash_bins]`` 고정 특징 행렬을 반환한다."""

    _validate_bins(hash_bins)
    if not texts:
        return np.empty((0, len(DENSE_FEATURE_NAMES) + hash_bins), dtype=float)
    return np.asarray(
        [_raw_feature_vector(str(text), hash_bins) for text in texts],
        dtype=float,
    )
