# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""설정 하나를 공개 Train+Dev로 재적합해 제출 산출물로 굳힌다.

    PYTHONPATH=src python3 tools/export_artifact.py \
        --config experiments/configs/c1-ridge-pertier.json

기본 출력 경로는 ``src/router/resources/artifact.v1.json``이며 패키지에 함께
실린다. 런타임은 이 파일을 읽기만 하고 학습하지 않는다.

설정 선택과 안전 보정은 Train/Dev를 분리해 끝낸다. 그 뒤에는 설정을 더 고르지
않고, 규칙상 학습에 허용된 두 공개 split을 모두 사용해 최종 계수만 재적합한다.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from router.artifact import build_artifact, write_artifact
from router.data import combine_datasets, load_dataset
from router.heads import build_cost_head, build_gate, build_score_head
from router.pipeline import Config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src" / "router" / "resources" / "artifact.v1.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_set_sha256(paths: tuple[Path, ...]) -> str:
    """파일 경로와 내용 digest의 고정 manifest digest를 만든다."""

    digest = hashlib.sha256()
    for path in paths:
        relative = str(path.resolve().relative_to(ROOT)).encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = Config.load(args.config)
    train_split = load_dataset("train")
    dev_split = load_dataset("dev")
    train = combine_datasets(train_split, dev_split)

    score_head = build_score_head(config.score)
    cost_head = build_cost_head(config.cost)
    gate = build_gate(config.gate)
    for head in (score_head, cost_head, gate):
        head.fit(train)

    input_paths = tuple(
        ROOT / "data" / "materialized" / split / "inputs.json"
        for split in ("train", "dev")
    )
    outcome_paths = tuple(
        ROOT / "data" / split / "outcomes.json" for split in ("train", "dev")
    )
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "config_path": str(args.config.resolve().relative_to(ROOT)),
        "fit_split": "public-train-dev",
        "fit_episodes": len(train),
        "fit_inputs_sha256": file_set_sha256(input_paths),
        "fit_outcomes_sha256": file_set_sha256(outcome_paths),
        "fit_sources": {
            split: {
                "inputs_sha256": file_sha256(input_paths[i]),
                "outcomes_sha256": file_sha256(outcome_paths[i]),
                "episodes": len((train_split, dev_split)[i]),
            }
            for i, split in enumerate(("train", "dev"))
        },
        "policy_id": train.policy.policy_id,
        "note": (
            "설정 선택을 Train/Dev 분리 검증으로 동결한 뒤 공개 Train 1,760 + "
            "Dev 880문항으로 최종 재적합했다. 프롬프트 원문·문항 ID·문항별 "
            "선택은 저장하지 않는다."
        ),
    }

    artifact = build_artifact(
        config, score_head, cost_head, gate, provenance=provenance
    )
    write_artifact(args.output, artifact)

    size = args.output.stat().st_size
    print(f"산출물: {args.output.relative_to(ROOT)}  {size / 1024:.1f} KB")
    print(f"  설정 {config.id}  커밋 {provenance['commit'][:7]}")
    print(f"  SHA-256 {file_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
