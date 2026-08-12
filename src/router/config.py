# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""실험 설정과 안전 손잡이. **런타임 안전 모듈이다.**

제출 이미지 안에서 import되므로 평가 하네스나 학습 자료에 의존하면 안 된다.
여기서 ossp_router.scoring이나 router.harness를 부르는 순간 컨테이너에서
죽는다. 계약은 "설정을 읽고 사용률을 계산한다"까지다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from .data import TIERS

DEFAULT_UTIL = {"fast": 0.90, "balanced": 0.90, "premium": 0.85}


@dataclass(frozen=True)
class Config:
    id: str
    score: Any = "family"
    cost: Any = "family"
    gate: Any = "none"
    alloc: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @staticmethod
    def load(path: Path) -> "Config":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(raw) - {"id", "score", "cost", "gate", "alloc", "note"}
        if unknown:
            raise ValueError(f"알 수 없는 설정 키: {sorted(unknown)}")
        return Config(**raw)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "cost": self.cost,
            "gate": self.gate,
            "alloc": self.alloc,
            "note": self.note,
        }

    @property
    def util(self) -> Dict[str, float]:
        raw = self.alloc.get("util", DEFAULT_UTIL)
        if isinstance(raw, (int, float)):
            return {t: float(raw) for t in TIERS}
        merged = dict(DEFAULT_UTIL)
        merged.update({k: float(v) for k, v in raw.items()})
        return merged

    @property
    def mu(self) -> float:
        return float(self.alloc.get("mu", 0.0))

    @property
    def headroom(self) -> Dict[str, float] | None:
        """여윳돈을 얼마나 쓸지. **등급 간 비교가 되는 유일한 안전 손잡이다.**

        ``util``은 전체 예산 대비 비율이라 같은 값이 등급마다 전혀 다른 뜻이
        된다. ``util=0.9``면 Fast는 여윳돈 0.25 중 0.125(50%)를 쓰지만
        Premium은 3.0 중 2.6(87%)을 쓴다. 그래서 util에 일률적으로 -0.03을
        빼면 Fast만 여유가 통째로 사라진다(실측: Fast가 all-light로 퇴화).

        ``headroom=h``는 세 등급 모두 여윳돈의 h를 쓴다는 뜻이다.
        실효 배율은 ``1 + h * (배율 - 1)``이 된다.
        """

        raw = self.alloc.get("headroom")
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            return {t: float(raw) for t in TIERS}
        merged = {t: 0.85 for t in TIERS}
        merged.update({k: float(v) for k, v in raw.items()})
        return merged

    @property
    def size_penalty(self) -> float:
        """배치가 작을수록 여유를 더 준다.

        실현 비용 비율의 흔들림은 표본 수의 제곱근에 반비례한다. 200문항
        배치에서 파산이 몰리는 것이 그 때문이다. 배치 구성은 규칙상 볼 수 있는
        정보이고, 문항 ID나 입력 순서를 쓰는 것과는 다르다.
        """

        return float(self.alloc.get("size_penalty", 0.0))


def effective_util(
    config: Config, n_episodes: int, multipliers: Mapping[str, float]
) -> Dict[str, float]:
    """할당기에 넘길 등급별 사용률을 만든다.

    ``headroom``이 지정되면 그것을 우선 쓰고, 없으면 예전 ``util``을 쓴다.
    반환값은 항상 '전체 예산 대비 비율'이라 할당기 쪽은 바뀌지 않는다.
    """

    penalty = config.size_penalty / max(1.0, float(n_episodes)) ** 0.5
    headroom = config.headroom
    if headroom is None:
        return {t: max(0.0, u - penalty) for t, u in config.util.items()}
    result = {}
    for tier, h in headroom.items():
        multiplier = float(multipliers[tier])
        adjusted = max(0.0, h - penalty)
        result[tier] = (1.0 + adjusted * (multiplier - 1.0)) / multiplier
    return result


