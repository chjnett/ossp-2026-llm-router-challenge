# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""학습 산출물 저장과 로드.

제출 이미지 안에서는 **학습을 하지 않는다.** 오프라인에서 Train으로 적합한
계수를 JSON 하나로 굳혀 이미지에 넣고, 런타임은 그것을 읽기만 한다.

JSON을 쓰는 이유는 크기가 아니라 **감사 가능성**이다. 심사자가 이미지 안에
무엇이 들어 있는지 그대로 볼 수 있어야 하고, 규칙상 포함 파일의 출처와
내용을 밝혀야 한다. 전체 26 KB 남짓이라 형식으로 손해 볼 것도 없다.

산출물에는 프롬프트 원문, 문항 ID, 문항별 특징이나 선택을 넣지 않는다.
전역 계수와 계열별 통계, 그리고 재현에 필요한 해시만 남긴다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .constants import MODEL_IDS
from .features import FAMILIES, FEATURE_NAMES, FEATURE_VERSION

ARTIFACT_SCHEMA_VERSION = 1


def build_artifact(
    config,
    score_head,
    cost_head,
    gate,
    *,
    provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """적합이 끝난 헤드들을 하나의 산출물로 묶는다."""

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "config": config.as_dict(),
        "models": list(MODEL_IDS),
        "features": {
            "version": FEATURE_VERSION,
            "names": list(FEATURE_NAMES),
            "families": list(FAMILIES),
        },
        "versions": {
            "score": score_head.version,
            "cost": cost_head.version,
            "gate": gate.version,
        },
        "state": {
            "score": score_head.state(),
            "cost": cost_head.state(),
            "gate": gate.state(),
        },
        "provenance": provenance or {},
    }


def write_artifact(path: Path, artifact: Dict[str, Any]) -> None:
    """원자적으로 쓴다. 부분 파일이 유효한 산출물로 보이면 안 된다."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_artifact(path: Path) -> Dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"산출물 schema_version이 {ARTIFACT_SCHEMA_VERSION}이 아니다: "
            f"{artifact.get('schema_version')!r}"
        )
    if artifact.get("models") != list(MODEL_IDS):
        raise ValueError("산출물의 모델 순서가 코드와 다르다")
    features = artifact.get("features", {})
    if features.get("version") != FEATURE_VERSION:
        raise ValueError(
            f"특징 추출기 버전이 다르다: 산출물 {features.get('version')!r} "
            f"vs 코드 {FEATURE_VERSION!r}"
        )
    if features.get("names") != list(FEATURE_NAMES):
        raise ValueError("특징 이름과 순서가 코드와 다르다. 계수를 그대로 쓸 수 없다")
    if features.get("families") != list(FAMILIES):
        raise ValueError("계열 목록이 코드와 다르다")
    return artifact


def restore(artifact: Dict[str, Any], policy):
    """산출물에서 세 헤드를 되살린다. 학습은 하지 않는다."""

    from .heads import build_cost_head, build_gate, build_score_head
    from .config import Config

    config = Config(**artifact["config"])
    score_head = build_score_head(config.score)
    cost_head = build_cost_head(config.cost)
    gate = build_gate(config.gate)

    score_head.load_state(artifact["state"]["score"])
    cost_head.load_state(artifact["state"]["cost"], policy)
    gate.load_state(artifact["state"]["gate"])

    for name, head in (("score", score_head), ("cost", cost_head), ("gate", gate)):
        expected = artifact["versions"][name]
        if head.version != expected:
            raise ValueError(
                f"{name} 헤드 버전 불일치: 산출물 {expected!r} vs 코드 {head.version!r}"
            )
    return config, score_head, cost_head, gate
