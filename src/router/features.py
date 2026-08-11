# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""[F] 프롬프트 내용만으로 계산하는 특징.

이 모듈은 텍스트 하나만 받는다. 문항 ID, 입력 순서, 등급, 출처, 평가 결과는
인자로 받을 수조차 없다 (RULES A1).

계열(family)은 프롬프트 내용에서 정규식으로 추론한다. 출처 이름을 메타데이터로
읽는 것은 금지지만, 프롬프트 본문에서 형태를 알아보는 것은 내용 기반 라우팅으로
명시적으로 허용된다 (CHALLENGE_RULES.md '사용할 수 있는 정보').
"""

from __future__ import annotations

import re
from typing import Sequence, Tuple

import numpy as np

# 계열은 비용·이득 구조가 크게 다른 덩어리로 나눈다.
# axk1-think의 출력토큰이 프롬프트 길이와 무상관(corr 0.02)이고 계열에만
# 의존하므로, 비용 예측의 뼈대가 곧 이 분류다.
FAMILIES: Tuple[str, ...] = (
    "code_io",     # 코드 실행 결과 예측
    "mcq_ko",      # 한국어 객관식
    "mcq_en",      # 영문 객관식
    "long_ctx",    # 장문 문맥
    "logic",       # 규칙 기반 논리 추론
    "sym_math",    # 기호·수식 계산
    "word_math",   # 문장제
    "ko_open",     # 한국어 서술형
    "other",
)
FAMILY_INDEX = {name: i for i, name in enumerate(FAMILIES)}

_CODE_FN = re.compile(r"\bdef\s+\w+\s*\(")
_CODE_ASSERT = re.compile(r"\bassert\b")
_OPTION_LINE = re.compile(r"^\s*(?:[A-D][\.\)]|\([A-D]\)|[①-⑩])\s", re.M)
_QUESTION_LINE = re.compile(r"^\s*(?:Question|질문|문제)\s*[:：]", re.M)
_HANGUL = re.compile(r"[가-힣]")
_LOGIC = re.compile(
    r"\b(?:is round|is kind|is big|is young|is white|is nice|is furry|"
    r"does it follow|true or false|if someone is)\b",
    re.I,
)
_MATH_OPEN = re.compile(
    r"^\s*(?:Let|Suppose|Solve|Calculate|Simplify|Find|Round|Differentiate|"
    r"Evaluate|Factor|Expand|Sort|Divide|Add|Multiply|Subtract|Convert|Put|"
    r"Collect|Work out|What is|Which is|Total of|Determine|Rearrange|Sum|Product)\b"
)
_MONEY = re.compile(r"\$|\bhow much\b|\bhow many\b|원\b", re.I)
_LATEX = re.compile(r"\\[a-zA-Z]+")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SENTENCE_END = re.compile(r"[.!?。！？]")
_WORD = re.compile(r"[A-Za-z가-힣]+")

LONG_CONTEXT_CHARS = 6_000


def family_of(text: str) -> str:
    """프롬프트 본문의 형태만 보고 계열을 고른다. 판정 순서가 곧 우선순위다."""

    if _CODE_FN.search(text) and _CODE_ASSERT.search(text):
        return "code_io"
    if len(text) >= LONG_CONTEXT_CHARS:
        return "long_ctx"
    has_hangul = bool(_HANGUL.search(text))
    looks_mcq = bool(_OPTION_LINE.search(text)) or bool(_QUESTION_LINE.search(text))
    if has_hangul:
        return "mcq_ko" if looks_mcq else "ko_open"
    # 규칙 기반 논리 추론은 본문 뒤에 'Question:' 줄을 달고 오는 경우가 많아
    # MCQ 판정보다 먼저 걸러야 한다. 이 순서를 바꾸면 K1 ROI가 7.0인 계열과
    # 0.8인 계열이 한 덩어리로 섞여 배차가 무너진다.
    if _LOGIC.search(text):
        return "logic"
    if looks_mcq:
        return "mcq_en"
    if _MATH_OPEN.search(text) and not _MONEY.search(text):
        return "sym_math"
    if _MONEY.search(text):
        return "word_math"
    return "other"


def family_codes(texts: Sequence[str]) -> np.ndarray:
    return np.array([FAMILY_INDEX[family_of(t)] for t in texts], dtype=int)


# 특징 이름을 고정한다. 순서가 바뀌면 캐시 키도 바뀌어야 한다.
FEATURE_NAMES: Tuple[str, ...] = (
    "log_chars",
    "log_words",
    "log_lines",
    "log_max_line",
    "log_sentences",
    "hangul_ratio",
    "digit_ratio",
    "log_numbers",
    "log_max_number",
    "paren_depth",
    "n_options",
    "n_questions",
    "latex_count",
    "code_markers",
    "indent_ratio",
    "type_token_ratio",
    "is_long_context",
)
FEATURE_VERSION = "f1"


def _row(text: str) -> list[float]:
    lines = text.split("\n")
    words = _WORD.findall(text)
    numbers = [abs(float(x)) for x in _NUMBER.findall(text)[:400]]
    nonspace = max(1, sum(not c.isspace() for c in text))

    depth = best = 0
    for ch in text:
        if ch in "([{":
            depth += 1
            best = max(best, depth)
        elif ch in ")]}":
            depth = max(0, depth - 1)

    return [
        np.log1p(len(text)),
        np.log1p(len(words)),
        np.log1p(len(lines)),
        np.log1p(max((len(l) for l in lines), default=0)),
        np.log1p(len(_SENTENCE_END.findall(text))),
        len(_HANGUL.findall(text)) / nonspace,
        sum(c.isdigit() for c in text) / nonspace,
        np.log1p(len(numbers)),
        np.log1p(max(numbers, default=0.0)),
        float(best),
        float(len(_OPTION_LINE.findall(text))),
        float(len(_QUESTION_LINE.findall(text))),
        float(len(_LATEX.findall(text))),
        float(len(_CODE_FN.findall(text)) + len(_CODE_ASSERT.findall(text))),
        len([l for l in lines if l[:2] == "  "]) / max(1, len(lines)),
        len(set(words)) / max(1, len(words)),
        float(len(text) >= LONG_CONTEXT_CHARS),
    ]


def extract(texts: Sequence[str]) -> np.ndarray:
    """[n, len(FEATURE_NAMES)] 실수 배열."""

    return np.asarray([_row(t) for t in texts], dtype=float)
