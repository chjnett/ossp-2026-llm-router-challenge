# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""``router-run`` — 평가에서 실제로 실행되는 진입점.

    router-run --input /challenge/input/inputs.json \
               --tier fast \
               --output /challenge/output/submission.json

규격은 ``docs/RUNTIME.md``를 따른다. 이 파일이 지켜야 하는 것:

* **어떤 예외도 등급을 죽이지 않는다** (RULES E1). 프로세스 시작 직후 all-light
  결정을 먼저 확보하고, 예측이 실패하면 그대로 낸다. 크래시는 등급 0점이다.
* **워치독** (E2). 등급별 90초 한도의 60%가 지나면 남은 계산을 접고 즉시 쓴다.
* **원자적 쓰기, 권한 0644, 출력은 submission.json 하나** (E3).
* **로그 최소화** (E4). stdout·stderr 각 1 MiB를 넘으면 실행 실패다.
* 학습은 하지 않는다. 이미지에 굳혀 넣은 산출물을 읽기만 한다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from ossp_router.protocol import (
    TIERS,
    Decision,
    ProtocolError,
    Submission,
    dumps_json,
    load_bundled_policy,
    load_input,
    parse_submission,
    submission_to_dict,
)

from .constants import MODEL_IDS

# 등급별 한도 90초의 60%. 남은 40%는 쓰기와 컨테이너 종료에 남긴다.
TIME_BUDGET_SECONDS = 54.0

ARTIFACT_PATH = Path(__file__).resolve().parent / "resources" / "artifact.v1.json"


def _all_light(episode_ids: Sequence[str]) -> list[str]:
    return [MODEL_IDS[0]] * len(episode_ids)


def _select(inputs, tier: str, deadline: float) -> list[str]:
    """산출물을 읽어 모델을 고른다. 실패하면 호출자가 all-light로 떨어뜨린다."""

    from .allocate import allocate
    from .artifact import load_artifact, restore
    from .constants import content_key, episode_text
    from .config import effective_util

    policy = load_bundled_policy()
    artifact = load_artifact(ARTIFACT_PATH)
    config, score_head, cost_head, gate = restore(artifact, policy)

    texts = [episode_text(episode) for episode in inputs.episodes]
    keys = [content_key(text) for text in texts]

    # 무거운 단계 사이마다 확인한다. 한 번만 보면 그 단계 안에서 늦어질 때
    # 못 잡는다. 2,640문항이 2초라 여유는 크지만, 비공개셋 크기를 모르므로
    # 각 구간이 끝날 때마다 남은 시간을 본다.
    def check(stage: str) -> None:
        if time.monotonic() > deadline:
            raise TimeoutError(f"시간 예산 초과: {stage}")

    check("산출물 로드")
    s_hat = score_head.predict(texts)
    check("점수 예측")
    c_hat, sd = cost_head.predict(texts)
    check("비용 예측")
    allow = gate.allow(texts, s_hat, c_hat)
    check("게이트")

    multipliers = {t: float(policy.tiers[t].budget_multiplier) for t in TIERS}
    util = effective_util(config, len(texts), multipliers)
    picks = allocate(
        s_hat,
        c_hat,
        multiplier=multipliers[tier],
        util=util[tier],
        allow=allow,
        sd=sd,
        mu=config.mu,
        keys=keys,
    ).picks
    return [MODEL_IDS[int(p)] for p in picks]


def _write_atomic(path: Path, submission: Submission) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            dumps_json(submission_to_dict(submission)), encoding="utf-8"
        )
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-run",
        description="프롬프트 내용과 등급만 보고 문항별 모델을 하나 고른다.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started = time.monotonic()
    deadline = started + TIME_BUDGET_SECONDS
    args = _parser().parse_args(argv)

    # 입력을 못 읽으면 낼 것이 없다. 규격대로 종료 코드 2다.
    try:
        inputs = load_input(args.input)
        policy = load_bundled_policy()
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    episode_ids = [episode.episode_id for episode in inputs.episodes]
    # 예측을 시작하기 전에 유효한 답을 손에 쥔다.
    chosen = _all_light(episode_ids)
    fallback = False

    try:
        chosen = _select(inputs, args.tier, deadline)
    except BaseException as exc:  # noqa: BLE001 - 어떤 실패도 등급을 죽이면 안 된다
        fallback = True
        print(f"경고: 폴백 사용 ({type(exc).__name__})", file=sys.stderr)

    try:
        submission = parse_submission(
            submission_to_dict(
                Submission(
                    schema_version=inputs.schema_version,
                    challenge_id=inputs.challenge_id,
                    policy_id=policy.policy_id,
                    split=inputs.split,
                    tier=args.tier,
                    decisions=tuple(
                        Decision(episode_id, model_id)
                        for episode_id, model_id in zip(episode_ids, chosen)
                    ),
                )
            )
        )
        _write_atomic(args.output, submission)
    except (OSError, ProtocolError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2

    print(
        f"OK {args.tier} n={len(episode_ids)} "
        f"{'fallback ' if fallback else ''}{time.monotonic() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
