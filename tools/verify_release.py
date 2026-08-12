# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""제출 직전 관문. `submission-ossp-skt.json`이 가리키는 것이 실제인지 본다.

공식 검증기(`tools/validate_technical_submission.py`)는 **형식만** 본다.
40자리 16진수이고 다이제스트 정규식에 맞으면 통과한다. 그래서 다음이 전부
조용히 통과한다.

* 커밋 SHA는 맞는데 이미지는 사흘 전 코드로 구운 것
* 이미지는 맞는데 커밋을 push하지 않아 심사자가 열 수 없는 것
* `repository_url`이 fork가 아니라 상류를 가리키는 것

산출물이 낡아 런타임이 조용히 all-light로 떨어진 사고가 세 번 있었다. 폴백도
유효한 제출을 만들기 때문에 형식 검사로는 안 잡혔다. 제출 메타데이터도 똑같이
"형식은 맞는데 내용이 틀린" 실패를 낸다. 여기서 내용을 확인한다.

    PYTHONPATH=src python3 tools/verify_release.py --json submission-ossp-skt.json

종료 코드는 통과 0, 검사 실패 1, 설정 오류 2다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

# 공식 한도. RUNTIME.md의 자원 한도 표.
COMPRESSED_LAYER_LIMIT = 1024 * 1024 * 1024
ROOTFS_LIMIT = 2 * 1024 * 1024 * 1024

SOURCE_MANIFEST_LABEL = "io.sktelecom.ossp.source-manifest-sha256"

# 이미지 안에서 파일 목록과 해시를 뽑는다. 라벨은 build arg로 아무 값이나
# 넣을 수 있으니 라벨만 믿지 않고 실제 내용도 대조한다.
_HASH_INSIDE = """
import hashlib, json, os
root = "/opt/router/router"
out = {}
for base, dirs, names in os.walk(root):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for name in names:
        if name.endswith((".pyc", ".pyo")):
            continue
        path = os.path.join(base, name)
        with open(path, "rb") as handle:
            out[os.path.relpath(path, root)] = hashlib.sha256(handle.read()).hexdigest()
print(json.dumps(out, sort_keys=True))
"""


class Failed(Exception):
    """검사가 실패했다."""


def _run(command: Sequence[str], **kwargs) -> str:
    result = subprocess.run(
        list(command), capture_output=True, text=True, check=False, **kwargs
    )
    if result.returncode != 0:
        raise Failed(
            f"명령이 실패했다: {' '.join(command[:4])}...\n{result.stderr.strip()[:500]}"
        )
    return result.stdout


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _export_commit(commit: str, destination: Path) -> None:
    """커밋 내용을 그대로 펼친다. 작업 트리가 아니라 커밋이 기준이다."""

    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if archive.returncode != 0:
        raise Failed(f"커밋 {commit[:12]}을 펼칠 수 없다: {archive.stderr.decode()[:300]}")
    with tempfile.NamedTemporaryFile(suffix=".tar") as handle:
        handle.write(archive.stdout)
        handle.flush()
        with tarfile.open(handle.name) as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk():
                    raise Failed(f"커밋에 링크가 있다: {member.name}")
                if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                    raise Failed(f"커밋에 수상한 경로가 있다: {member.name}")
            tar.extractall(destination)


def _source_manifest_of(tree: Path) -> str:
    """상류 정의를 그대로 쓴다. 우리가 따로 정의하면 값이 갈린다."""

    import benchmark_runtime

    original = benchmark_runtime.ROOT
    try:
        benchmark_runtime.ROOT = tree
        return str(benchmark_runtime._source_tree_manifest()["sha256"])
    finally:
        benchmark_runtime.ROOT = original


def _inspect(image: str, template: str) -> str:
    return _run(["docker", "image", "inspect", image, "--format", template]).strip()


# --------------------------------------------------------------------------
# 검사들. 각각 실패하면 Failed를 던진다.
# --------------------------------------------------------------------------


def check_tree_is_clean(_: dict) -> str:
    dirty = _run(["git", "status", "--porcelain"], cwd=ROOT).strip()
    if dirty:
        raise Failed(
            "작업 트리가 깨끗하지 않다. 커밋하지 않은 변경이 이미지에 들어갔을 수 있다:\n"
            + dirty[:400]
        )
    return "커밋하지 않은 변경 없음"


def check_schema(submission: dict) -> str:
    from validate_technical_submission import DEFAULT_SCHEMA, validate_submission

    schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
    validate_submission(submission, schema)
    return "공식 스키마 통과"


def check_repository_url(submission: dict) -> str:
    remote = _run(["git", "remote", "get-url", "origin"], cwd=ROOT).strip()
    normalized = remote.removesuffix(".git")
    declared = submission["repository_url"].removesuffix(".git").rstrip("/")
    if normalized != declared:
        raise Failed(f"repository_url이 origin과 다르다.\n  origin={normalized}\n  선언={declared}")
    upstream = _run(["git", "remote"], cwd=ROOT).split()
    if "upstream" in upstream:
        upstream_url = _run(["git", "remote", "get-url", "upstream"], cwd=ROOT).strip()
        if upstream_url.removesuffix(".git") == declared:
            raise Failed("repository_url이 참가자 fork가 아니라 상류를 가리킨다")
    return f"origin과 일치: {declared}"


def check_commit_is_public(submission: dict) -> str:
    commit = submission["commit_sha"]
    try:
        _run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT)
    except Failed as exc:
        raise Failed(f"커밋 {commit[:12]}이 저장소에 없다") from exc
    # 로컬 원격 추적 ref를 믿으면 안 된다. 이 저장소의 fetch refspec은
    # main으로 제한돼 있어 다른 브랜치의 origin/<이름>이 아예 생기지 않는다.
    # push해 둔 커밋을 "없다"고 잡았다. 원격에 직접 묻는다.
    #
    # 제출 커밋은 tip이 아니다. 코드 커밋 뒤에 JSON 커밋이 하나 더 오므로
    # 코드 커밋은 항상 부모다. tip 일치만 보면 안 되고 조상 판정이 필요하다.
    tips = []
    for line in _run(["git", "ls-remote", "--heads", "origin"], cwd=ROOT).splitlines():
        sha, _, ref = line.partition("\t")
        if sha.strip() and ref.strip():
            tips.append((sha.strip(), ref.strip()))
    if not tips:
        raise Failed("origin에서 브랜치 목록을 받지 못했다")

    for sha, ref in tips:
        if sha == commit:
            return f"origin에 있음 ({ref} tip)"
    for sha, ref in tips:
        _run(["git", "fetch", "--quiet", "origin", ref], cwd=ROOT)
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, sha],
            cwd=ROOT, capture_output=True, check=False,
        )
        if ancestor.returncode == 0:
            return f"origin에 있음 ({ref}의 조상)"
    raise Failed(
        f"커밋 {commit[:12]}이 origin의 어떤 브랜치에서도 닿지 않는다. "
        f"심사자가 열 수 없다. 먼저 push해야 한다 (확인한 브랜치 {len(tips)}개)"
    )


def check_image_digest_resolves(submission: dict) -> str:
    image = submission["image_digest"]
    digests = json.loads(_inspect(image, "{{json .RepoDigests}}"))
    if image not in digests:
        raise Failed(f"이미지가 그 다이제스트로 보이지 않는다. RepoDigests={digests}")
    architecture = _inspect(image, "{{.Architecture}}")
    operating_system = _inspect(image, "{{.Os}}")
    if (operating_system, architecture) != ("linux", "arm64"):
        raise Failed(f"공식 플랫폼은 linux/arm64인데 {operating_system}/{architecture}다")
    return f"{operating_system}/{architecture}, 다이제스트로 확인"


def check_no_declared_volume(submission: dict) -> str:
    """공식 사전 검사가 VOLUME 선언을 거부한다 (runtime.validate_image_configuration)."""

    from ossp_router.runtime import validate_image_configuration

    metadata = json.loads(_run(["docker", "image", "inspect", submission["image_digest"]]))[0]
    validate_image_configuration(metadata)
    return "VOLUME 선언 없음"


def check_image_carries_the_commit(submission: dict) -> str:
    """이 제출의 핵심. 이미지 안 코드가 제출 커밋의 코드와 같은가."""

    commit = submission["commit_sha"]
    image = submission["image_digest"]

    # Go 템플릿은 작은따옴표를 문자 상수로 읽는다. 큰따옴표여야 한다.
    label = _inspect(image, f'{{{{index .Config.Labels "{SOURCE_MANIFEST_LABEL}"}}}}')
    if not label or label in {"unbound", "<no value>"}:
        raise Failed(
            "이미지에 소스 매니페스트 라벨이 없다. "
            "--build-arg SOURCE_MANIFEST_SHA256=... 없이 빌드했다"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "commit"
        tree.mkdir()
        _export_commit(commit, tree)
        expected = _source_manifest_of(tree)
        if label != expected:
            raise Failed(
                "이미지 라벨의 소스 매니페스트가 제출 커밋과 다르다. "
                "이미지가 다른 코드로 구워졌다.\n"
                f"  이미지 라벨 = {label}\n"
                f"  커밋 {commit[:12]} = {expected}"
            )

        # 라벨은 build arg라 아무 값이나 넣을 수 있다. 실제 내용도 대조한다.
        inside = json.loads(
            _run(
                [
                    "docker", "run", "--rm", "--network", "none",
                    "--entrypoint", "python3", image, "-c", _HASH_INSIDE,
                ]
            )
        )
        source = tree / "src" / "router"
        outside = {}
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.name.endswith((".pyc", ".pyo")):
                continue
            outside[str(path.relative_to(source))] = _sha256_bytes(path.read_bytes())

        # 동일성이 아니라 부분집합이어야 한다. .dockerignore가 런타임 모듈만
        # 허용목록으로 싣고 학습·실험 코드(cache/harness/stress)는 일부러
        # 뺀다. 이미지에 없는 것은 정상이고, 이미지에 있는데 커밋에 없거나
        # 내용이 다른 것이 사고다.
        only_image = sorted(set(inside) - set(outside))
        changed = sorted(k for k in set(inside) & set(outside) if inside[k] != outside[k])
        if only_image or changed:
            raise Failed(
                "이미지 안 /opt/router/router가 커밋의 src/router에서 오지 않았다.\n"
                f"  커밋에 없는 파일이 이미지에 있음: {only_image[:5]}\n"
                f"  내용이 다름: {changed[:5]}"
            )
        if not inside:
            raise Failed("이미지에 라우터 파일이 하나도 없다")
        excluded = len(set(outside) - set(inside))

    return (
        f"이미지의 {len(inside)}개 파일이 커밋 {commit[:12]}과 바이트 단위로 일치 "
        f"(학습 전용 {excluded}개는 .dockerignore가 제외)"
    )


def check_artifact_is_not_stale(submission: dict) -> str:
    """이미지 안 산출물이 그 이미지 안 코드와 맞물리는가.

    이것이 실패하면 런타임이 조용히 all-light로 떨어진다. 세 번 겪었다.
    """

    script = (
        "from ossp_router.protocol import load_bundled_policy;"
        "from router.artifact import load_artifact, restore;"
        "from router.cli import ARTIFACT_PATH;"
        "restore(load_artifact(ARTIFACT_PATH), load_bundled_policy());"
        "print('ok')"
    )
    output = _run(
        [
            "docker", "run", "--rm", "--network", "none",
            "--entrypoint", "python3", submission["image_digest"], "-c", script,
        ]
    )
    if "ok" not in output:
        raise Failed("이미지 안에서 산출물과 코드가 맞물리지 않는다")
    return "산출물과 코드가 맞물린다 (폴백 아님)"


def check_size_evidence(submission: dict, evidence_path: Path | None) -> str:
    if evidence_path is None:
        return "건너뜀 (--evidence 없음)"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("submitted_digest") != submission["image_digest"]:
        raise Failed("크기 증거가 다른 이미지를 측정한 것이다")
    compressed = int(evidence["oci_compressed_layer_bytes"])
    rootfs = int(evidence["rootfs_apparent_bytes"])
    if compressed > COMPRESSED_LAYER_LIMIT:
        raise Failed(f"압축 계층 합계 {compressed:,}가 한도를 넘는다")
    if rootfs > ROOTFS_LIMIT:
        raise Failed(f"rootfs {rootfs:,}가 한도를 넘는다")
    return (
        f"압축 {compressed / 1048576:.1f} MiB (한도 1024), "
        f"rootfs {rootfs / 1048576:.1f} MiB (한도 2048)"
    )


CHECKS: tuple[tuple[str, Callable[[dict], str]], ...] = (
    ("작업 트리가 깨끗하다", check_tree_is_clean),
    ("공식 스키마를 통과한다", check_schema),
    ("repository_url이 참가자 fork다", check_repository_url),
    ("커밋이 origin에 공개되어 있다", check_commit_is_public),
    ("이미지가 그 다이제스트로 존재한다", check_image_digest_resolves),
    ("이미지에 VOLUME 선언이 없다", check_no_declared_volume),
    ("이미지가 그 커밋의 코드를 담고 있다", check_image_carries_the_commit),
    ("이미지 안 산출물이 낡지 않았다", check_artifact_is_not_stale),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="제출 메타데이터가 가리키는 커밋과 이미지가 실제로 맞물리는지 확인한다."
    )
    parser.add_argument("--json", type=Path, default=ROOT / "submission-ossp-skt.json")
    parser.add_argument(
        "--evidence", type=Path, help="router-measure-image가 만든 크기 증거"
    )
    args = parser.parse_args(argv)

    try:
        submission = json.loads(args.json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"오류: {args.json}을 읽을 수 없다: {exc}", file=sys.stderr)
        return 2

    checks = list(CHECKS) + [
        ("이미지 크기가 한도 안이다", lambda s: check_size_evidence(s, args.evidence))
    ]

    failures = 0
    for index, (name, check) in enumerate(checks, start=1):
        try:
            detail = check(submission)
        except Failed as exc:
            failures += 1
            print(f"[{index}/{len(checks)}] ✗ {name}\n      {exc}")
        except Exception as exc:  # noqa: BLE001 - 도구 오류도 통과로 오해하면 안 된다
            failures += 1
            print(f"[{index}/{len(checks)}] ✗ {name}\n      검사 자체가 실패했다: {exc!r}")
        else:
            print(f"[{index}/{len(checks)}] ✓ {name} — {detail}")

    print()
    if failures:
        print(f"{failures}건 실패. 제출하면 안 된다.")
        return 1
    print("전부 통과. 제출 메타데이터가 실제와 맞물린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
