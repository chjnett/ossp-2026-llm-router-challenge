# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""제출 관문이 실제로 막는지 확인한다.

항상 통과하는 관문은 없는 것보다 나쁘다. 통과를 확인해 주는 것처럼 보이면서
아무것도 안 막기 때문이다. 실제로 리허설에서 관문의 두 검사가 조용히
못 돌고 있었다(Go 템플릿 따옴표, 낡은 원격 추적 ref). 그래서 여기서는
**틀린 입력을 넣고 반드시 실패하는지**를 못 박는다.

Docker가 필요한 검사는 별도로 표시하고, 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_release as release  # noqa: E402

VALID = {
    "schema_version": 1,
    "challenge_id": "ossp-2026-llm-router-challenge",
    "repository_url": "https://github.com/chjnett/ossp-2026-llm-router-challenge",
    "commit_sha": "0" * 40,
    "image_digest": "registry.example.com/team/router@sha256:" + "0" * 64,
    "primary_license": "Apache-2.0",
}


def has_docker() -> bool:
    try:
        return (
            subprocess.run(
                ["docker", "version"], capture_output=True, timeout=30, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.SubprocessError):
        return False


class SchemaTest(unittest.TestCase):
    def test_accepts_a_well_formed_submission(self) -> None:
        self.assertEqual("공식 스키마 통과", release.check_schema(dict(VALID)))

    def test_rejects_a_short_commit_sha(self) -> None:
        broken = dict(VALID, commit_sha="abc123")
        with self.assertRaises(Exception):
            release.check_schema(broken)

    def test_rejects_a_tag_instead_of_a_digest(self) -> None:
        broken = dict(VALID, image_digest="registry.example.com/team/router:latest")
        with self.assertRaises(Exception):
            release.check_schema(broken)

    def test_rejects_an_extra_field(self) -> None:
        broken = dict(VALID, note="안녕")
        with self.assertRaises(Exception):
            release.check_schema(broken)


class RepositoryUrlTest(unittest.TestCase):
    def test_rejects_the_upstream_repository(self) -> None:
        """상류를 가리키면 참가자 fork가 아니다. 심사 대상이 잘못된다."""

        upstream = "https://github.com/sktelecom/ossp-2026-llm-router-challenge"
        with self.assertRaises(release.Failed):
            release.check_repository_url(dict(VALID, repository_url=upstream))

    def test_rejects_a_url_that_is_not_our_origin(self) -> None:
        with self.assertRaises(release.Failed):
            release.check_repository_url(
                dict(VALID, repository_url="https://github.com/someone/else")
            )

    def test_accepts_origin_with_or_without_dot_git(self) -> None:
        origin = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not origin:
            self.skipTest("origin 원격이 없다")
        base = origin.removesuffix(".git")
        for variant in (base, base + ".git", base + "/"):
            with self.subTest(variant=variant):
                release.check_repository_url(dict(VALID, repository_url=variant))


class CommitTest(unittest.TestCase):
    def test_rejects_a_commit_that_does_not_exist(self) -> None:
        with self.assertRaises(release.Failed):
            release.check_commit_is_public(dict(VALID, commit_sha="0" * 40))

    def test_accepts_a_commit_that_is_an_ancestor_of_a_remote_branch(self) -> None:
        """제출 커밋은 tip이 아니다. 뒤에 JSON 커밋이 하나 더 온다.

        tip 일치만 보면 실제 제출을 통째로 놓친다.
        """

        head = subprocess.run(
            ["git", "rev-parse", "HEAD~1"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
        if not head:
            self.skipTest("부모 커밋이 없다")
        try:
            detail = release.check_commit_is_public(dict(VALID, commit_sha=head))
        except release.Failed as exc:
            self.skipTest(f"원격에 접근할 수 없다: {exc}")
        self.assertIn("origin에 있음", detail)


class SizeEvidenceTest(unittest.TestCase):
    def _evidence(self, tmp: Path, **overrides) -> Path:
        payload = {
            "submitted_digest": VALID["image_digest"],
            "oci_compressed_layer_bytes": 66_453_790,
            "rootfs_apparent_bytes": 205_294_576,
        }
        payload.update(overrides)
        path = tmp / "evidence.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_accepts_measurements_inside_both_limits(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            detail = release.check_size_evidence(dict(VALID), self._evidence(Path(tmp)))
        self.assertIn("한도 1024", detail)

    def test_rejects_evidence_measured_from_a_different_image(self) -> None:
        """다른 이미지를 측정한 증거를 붙이면 크기 검사가 무의미해진다."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(
                Path(tmp),
                submitted_digest="registry.example.com/team/other@sha256:" + "1" * 64,
            )
            with self.assertRaises(release.Failed):
                release.check_size_evidence(dict(VALID), evidence)

    def test_rejects_layers_over_the_limit(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(
                Path(tmp), oci_compressed_layer_bytes=release.COMPRESSED_LAYER_LIMIT + 1
            )
            with self.assertRaises(release.Failed):
                release.check_size_evidence(dict(VALID), evidence)

    def test_rejects_rootfs_over_the_limit(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            evidence = self._evidence(
                Path(tmp), rootfs_apparent_bytes=release.ROOTFS_LIMIT + 1
            )
            with self.assertRaises(release.Failed):
                release.check_size_evidence(dict(VALID), evidence)


class SourceManifestTest(unittest.TestCase):
    def test_manifest_of_head_matches_the_upstream_tool(self) -> None:
        """우리가 따로 정의하면 값이 갈린다. 상류 정의를 그대로 써야 한다.

        작업 트리가 더러우면 두 값은 당연히 다르다. 실제로 산출물을 다시
        구우면 provenance의 커밋과 시각이 바뀌어 매니페스트가 달라진다.
        그 상태로 이미지를 빌드하면 라벨이 커밋과 어긋난다.
        """

        import tempfile

        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        ).stdout
        tracked = [line for line in dirty.splitlines() if not line.startswith("??")]
        if tracked:
            self.skipTest(f"작업 트리가 더럽다: {tracked[:3]}")

        expected = subprocess.run(
            [sys.executable, "tools/benchmark_runtime.py", "--print-source-manifest-sha256"],
            cwd=ROOT, capture_output=True, text=True, check=False,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        if not expected:
            self.skipTest("상류 도구를 실행할 수 없다")
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "commit"
            tree.mkdir()
            release._export_commit("HEAD", tree)
            self.assertEqual(expected, release._source_manifest_of(tree))

    def test_manifest_changes_when_a_source_file_changes(self) -> None:
        """내용이 바뀌었는데 값이 같으면 결속이 아무것도 증명하지 못한다."""

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "commit"
            tree.mkdir()
            release._export_commit("HEAD", tree)
            before = release._source_manifest_of(tree)
            target = tree / "src" / "router" / "constants.py"
            target.write_text(
                target.read_text(encoding="utf-8") + "\n# 한 줄\n", encoding="utf-8"
            )
            self.assertNotEqual(before, release._source_manifest_of(tree))


@unittest.skipUnless(has_docker(), "Docker가 없다")
class ImageTest(unittest.TestCase):
    def test_rejects_an_image_that_does_not_exist(self) -> None:
        missing = "localhost:5001/nope@sha256:" + "9" * 64
        with self.assertRaises(release.Failed):
            release.check_image_digest_resolves(dict(VALID, image_digest=missing))


if __name__ == "__main__":
    unittest.main()
