# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""설정 하나를 Train으로 적합해 이미지에 넣을 산출물로 굳힌다.

    PYTHONPATH=src python3 tools/export_artifact.py \
        --config experiments/configs/c1-ridge-pertier.json

기본 출력 경로는 ``src/router/resources/artifact.v1.json``이며 패키지에 함께
실린다. 런타임은 이 파일을 읽기만 하고 학습하지 않는다.

Dev는 적합에 쓰지 않는다 (RULES D2). Dev를 섞으면 보정용 자료가 사라진다.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from router.artifact import build_artifact, write_artifact
from router.data import load_dataset
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
    train = load_dataset("train")

    score_head = build_score_head(config.score)
    cost_head = build_cost_head(config.cost)
    gate = build_gate(config.gate)
    for head in (score_head, cost_head, gate):
        head.fit(train)

    train_inputs = ROOT / "data" / "materialized" / "train" / "inputs.json"
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_sha(),
        "config_path": str(args.config.resolve().relative_to(ROOT)),
        "fit_split": "train",
        "fit_episodes": len(train),
        "fit_inputs_sha256": file_sha256(train_inputs),
        "fit_outcomes_sha256": file_sha256(ROOT / "data" / "train" / "outcomes.json"),
        "policy_id": train.policy.policy_id,
        "note": (
            "Train 1,760문항으로만 적합했다. Dev는 보정 전용이라 섞지 않는다. "
            "프롬프트 원문·문항 ID·문항별 선택은 저장하지 않는다."
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
