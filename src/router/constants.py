# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""제3자 의존성이 하나도 없는 최소 상수와 도우미.

**폴백 경로가 이 모듈만으로 굴러가야 한다.** numpy가 없거나 산출물이 깨져도
all-light 제출은 나와야 하고, 그러려면 진입점이 무거운 모듈을 import 시점에
끌어오면 안 된다. 실제로 numpy 없는 환경에서 ``router-run``이 import 단계에서
죽는 것을 상류 wheel 테스트가 잡았다.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

# 싼 것부터. argmax 동률을 이 순서로 결정적으로 깬다 (RULES B3).
MODEL_IDS: Tuple[str, ...] = ("ax31-light", "ax31", "axk1-think")
LIGHT = MODEL_IDS[0]
TIERS: Tuple[str, ...] = ("fast", "balanced", "premium")


def episode_text(episode) -> str:
    """라우팅 시점에 볼 수 있는 텍스트만 반환한다."""

    if episode.prompt is not None:
        return episode.prompt
    return "\n".join(message.content for message in episode.messages or ())


def content_key(text: str) -> str:
    """프롬프트 내용에서만 유도한 고정 키.

    내장 ``hash()``는 PYTHONHASHSEED에 따라 프로세스마다 달라지므로 쓰지 않는다
    (RULES B4).
    """

    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
