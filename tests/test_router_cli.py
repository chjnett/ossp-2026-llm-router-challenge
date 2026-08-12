# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""``router-run`` 런타임 — 평가에서 실제로 실행되는 경로.

여기서 실패하면 등급이 통째로 0점이거나 자동 채점이 멈춘다. Docker 없이도
확인할 수 있는 것들을 전부 못 박는다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

from ossp_router.protocol import load_input, load_submission  # noqa: E402

from router.cli import ARTIFACT_PATH, main  # noqa: E402

TOY = ROOT / "data" / "toy" / "inputs.json"
DEV = ROOT / "data" / "materialized" / "dev" / "inputs.json"


def run_cli(input_path: Path, tier: str, out: Path) -> int:
    return main(
        ["--input", str(input_path), "--tier", tier, "--output", str(out)]
    )


def decisions(path: Path) -> dict[str, str]:
    submission = load_submission(path)
    return {d.episode_id: d.model_id for d in submission.decisions}


class ArtifactTest(unittest.TestCase):
    def test_artifact_ships_with_the_package(self) -> None:
        self.assertTrue(
            ARTIFACT_PATH.exists(),
            "학습 산출물이 없다. tools/export_artifact.py를 먼저 돌려야 한다",
        )

    def test_artifact_state_holds_numbers_only(self) -> None:
        """학습 산출물에는 계수만 들어가야 한다.

        프롬프트 원문·문항 ID·문항별 선택이 섞이면 규칙 위반이다. 파일 전체
        문자열 검색은 설명문에도 걸리므로, ``state`` 안에 **문자열 값이
        하나도 없는지**로 정확히 검사한다. 계열 번호 같은 사전 키는 허용한다.
        """

        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

        def walk(node, path="state"):
            if isinstance(node, dict):
                for key, value in node.items():
                    self.assertRegex(
                        str(key), r"^[A-Za-z0-9_]+$", f"{path}: 수상한 키 {key!r}"
                    )
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for i, value in enumerate(node):
                    walk(value, f"{path}[{i}]")
            else:
                self.assertIsInstance(
                    node, (int, float), f"{path}: 숫자가 아닌 값 {node!r}"
                )

        walk(artifact["state"])
        self.assertLess(
            ARTIFACT_PATH.stat().st_size, 512 * 1024, "산출물이 예상보다 크다"
        )

    def test_artifact_records_provenance(self) -> None:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        provenance = artifact["provenance"]
        for key in ("commit", "fit_split", "fit_inputs_sha256", "policy_id"):
            self.assertIn(key, provenance)
        self.assertEqual("train", provenance["fit_split"], "Dev로 적합하면 안 된다")


class RuntimeTest(unittest.TestCase):
    def test_produces_a_valid_submission_for_every_tier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for tier in ("fast", "balanced", "premium"):
                out = Path(tmp) / f"{tier}.json"
                self.assertEqual(0, run_cli(TOY, tier, out))
                submission = load_submission(out)
                self.assertEqual(tier, submission.tier)
                self.assertEqual(3, len(submission.decisions))
                self.assertEqual(0o644, out.stat().st_mode & 0o777)

    def test_output_directory_holds_only_the_submission(self) -> None:
        """출력 볼륨 루트에 다른 파일이 남으면 운영자 검사에 걸린다."""

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "submission.json"
            self.assertEqual(0, run_cli(TOY, "fast", out))
            self.assertEqual(["submission.json"], sorted(p.name for p in Path(tmp).iterdir()))

    def test_bad_input_path_exits_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = run_cli(Path(tmp) / "없는파일.json", "fast", Path(tmp) / "o.json")
            self.assertEqual(2, code)

    def test_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            run_cli(DEV, "balanced", a)
            run_cli(DEV, "balanced", b)
            self.assertEqual(a.read_bytes(), b.read_bytes(), "같은 입력에 결과가 다르다")

    def test_id_and_order_audit(self) -> None:
        """평가에서 ID를 전부 바꾸고 순서를 섞은 입력으로 한 번 더 돌린다.

        선택이 하나라도 다르면 자동 채점이 멈춘다 (OPERATIONS.md).
        """

        import random

        original = json.loads(DEV.read_text(encoding="utf-8"))
        episodes = original["episodes"]
        rng = random.Random(20260812)
        shuffled = list(episodes)
        rng.shuffle(shuffled)
        # 어떤 문항도 원래 ID를 유지하지 않게 재배정한다.
        new_ids = [f"audit-{i:06d}" for i in range(len(shuffled))]
        mapping = {}
        audited = []
        for new_id, episode in zip(new_ids, shuffled):
            copy = dict(episode)
            mapping[new_id] = episode["episode_id"]
            copy["episode_id"] = new_id
            audited.append(copy)

        with tempfile.TemporaryDirectory() as tmp:
            audit_input = Path(tmp) / "audit.json"
            audit_input.write_text(
                json.dumps({**original, "episodes": audited}, ensure_ascii=False),
                encoding="utf-8",
            )
            for tier in ("fast", "balanced", "premium"):
                plain = Path(tmp) / f"{tier}-plain.json"
                audit = Path(tmp) / f"{tier}-audit.json"
                run_cli(DEV, tier, plain)
                run_cli(audit_input, tier, audit)
                base = decisions(plain)
                moved = {
                    mapping[k]: v for k, v in decisions(audit).items()
                }
                self.assertEqual(
                    base, moved, f"{tier}: ID·순서를 바꾸자 선택이 달라졌다"
                )

    def test_finishes_well_inside_the_time_limit(self) -> None:
        """공개 Dev 전체가 30초 안에 끝나야 한다 (RULES E5, 한도 90초)."""

        import time

        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            run_cli(DEV, "premium", Path(tmp) / "o.json")
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 30.0, f"{elapsed:.1f}초 걸렸다")


class FallbackTest(unittest.TestCase):
    def test_falls_back_to_all_light_when_the_artifact_is_broken(self) -> None:
        """산출물이 깨져도 유효한 제출이 나와야 한다. 크래시는 등급 0점이다."""

        backup = ARTIFACT_PATH.read_bytes()
        try:
            ARTIFACT_PATH.write_text("{ 깨진 JSON", encoding="utf-8")
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "o.json"
                self.assertEqual(0, run_cli(TOY, "premium", out))
                picks = set(decisions(out).values())
                self.assertEqual({"ax31-light"}, picks)
        finally:
            ARTIFACT_PATH.write_bytes(backup)

    def test_entry_point_is_wired_to_our_router(self) -> None:
        """setup.cfg의 router-run이 baseline이 아니라 우리 라우터를 가리켜야 한다."""

        text = (ROOT / "setup.cfg").read_text(encoding="utf-8")
        self.assertIn("router-run = router.cli:main", text)


class ContainerReadinessTest(unittest.TestCase):
    def test_runtime_imports_do_not_touch_training_data(self) -> None:
        """런타임 경로가 data/ 를 읽으면 이미지 안에서 죽는다."""

        script = (
            "import sys; sys.path.insert(0, %r);\n"
            "import router.cli as c;\n"
            "import inspect;\n"
            "src = inspect.getsource(c);\n"
            "assert 'load_dataset' not in src, 'cli가 학습 자료 로더를 부른다';\n"
            "print('ok')\n" % str(SRC)
        )
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(0, out.returncode, out.stderr)


if __name__ == "__main__":
    unittest.main()
