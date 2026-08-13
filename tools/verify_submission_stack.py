# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""제출 스택 전체를 한 번에 검증한다.

    PYTHONPATH=src python3 tools/verify_submission_stack.py [--rebuild]

챔피언 설정 → 산출물 → 이미지 → 컨테이너 실행 → 공식 채점까지가 **하나라도
어긋나면** 실패한다. 이 검사를 만든 이유는 같은 사고를 세 번 냈기 때문이다.
헤드 버전을 올리고 산출물을 다시 굽지 않으면 런타임이 조용히 all-light로
떨어지는데, 폴백도 유효한 제출을 만들어서 형식 검사로는 안 잡힌다.

검사 항목
  1. 산출물이 챔피언 설정에서 나왔는가
  2. 산출물과 현재 코드가 맞물리는가 (restore가 성공하는가)
  3. 호스트 실행이 폴백 없이 동작하는가
  4. 컨테이너 실행이 폴백 없이 동작하는가
  5. 호스트와 컨테이너의 선택이 **완전히 같은가** (numpy 버전이 다르다)
  6. ID를 전부 바꾸고 순서를 섞어도 컨테이너 선택이 같은가 (감사 재실행)
  7. 컨테이너 출력을 공식 채점기가 통과시키는가
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "src" / "router" / "resources" / "artifact.v1.json"
CHAMPION = ROOT / "experiments" / "champion.json"
DEV_INPUT = ROOT / "data" / "materialized" / "dev" / "inputs.json"
IMAGE = "ossp-router:verify"
TIERS = ("fast", "balanced", "premium")

ISOLATION = [
    "--rm", "--network", "none", "--read-only",
    "--cpus", "2", "--memory", "2g", "--memory-swap", "2g",
    "--pids-limit", "32", "--user", "65532:65532", "--ipc", "none",
    "--tmpfs", "/tmp:size=256m",
]


def fail(message: str) -> None:
    print(f"  ✗ {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"  ✓ {message}")


def decisions(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {d["episode_id"]: d["model_id"] for d in data["decisions"]}


def run_container(input_dir: Path, input_name: str, out_dir: Path, tier: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["docker", "run", *ISOLATION,
         "-v", f"{input_dir}:/in:ro", "-v", f"{out_dir}:/out",
         IMAGE, "--input", f"/in/{input_name}", "--tier", tier,
         "--output", f"/out/{tier}.json"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        fail(f"컨테이너 {tier} 실행 실패: {result.stderr[:300]}")
    return result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="산출물과 이미지를 다시 만든다")
    # 공식 플랫폼은 linux/arm64다. 다만 CI 러너는 amd64라 QEMU 없이는 못 굽고,
    # 에뮬레이션은 느리다. 여기서 확인하는 것(산출물-코드 정합, 폴백 없음,
    # 호스트-컨테이너 일치, ID·순서 감사)은 **플랫폼과 무관**하므로 CI에서는
    # 네이티브로 굽는다. arm64 빌드는 제출 리허설에서 따로 확인한다.
    parser.add_argument(
        "--platform", default="linux/arm64",
        help="이미지 플랫폼 (기본 linux/arm64, CI는 native)",
    )
    args = parser.parse_args()

    if not CHAMPION.exists():
        fail("챔피언이 없다")
    champion = json.loads(CHAMPION.read_text(encoding="utf-8"))
    config_id = champion["config"]["id"]
    print(f"챔피언: {config_id}\n")

    if args.rebuild:
        config_path = ROOT / "experiments" / "configs" / f"{config_id}.json"
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "export_artifact.py"),
             "--config", str(config_path)],
            cwd=ROOT, check=True, capture_output=True,
        )
        ok("산출물 재생성")
        # 소스 매니페스트 라벨을 붙여 굽는다. 이것을 빼면 이 도구가 만든
        # 이미지는 tools/verify_release.py가 검사할 수 없는 물건이 된다.
        # 같은 Dockerfile을 두 도구가 다르게 굽고 있으면 안 된다.
        manifest = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "benchmark_runtime.py"),
             "--print-source-manifest-sha256"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
        if manifest.returncode != 0:
            fail(f"소스 매니페스트를 계산하지 못했다: {manifest.stderr[-300:]}")

        command = ["docker", "build"]
        if args.platform != "native":
            command += ["--platform", args.platform]
        command += [
            "--provenance=false", "--sbom=false",
            "--build-arg", f"SOURCE_MANIFEST_SHA256={manifest.stdout.strip()}",
            "--file", "container/router.Dockerfile", "--tag", IMAGE, ".",
        ]
        build = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=1800,
        )
        if build.returncode != 0:
            fail(f"이미지 빌드 실패: {build.stderr[-300:]}")
        ok(f"이미지 빌드 ({args.platform})")

    # 1. 산출물이 챔피언 설정에서 나왔는가
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if artifact["config"]["id"] != config_id:
        fail(f"산출물이 다른 설정에서 나왔다: {artifact['config']['id']} != {config_id}")
    ok(f"산출물이 챔피언 설정에서 나왔다 ({ARTIFACT.stat().st_size / 1024:.0f} KB)")

    # 2. 산출물과 코드가 맞물리는가
    sys.path.insert(0, str(ROOT / "src"))
    from ossp_router.protocol import load_bundled_policy  # noqa: E402
    from router.artifact import load_artifact, restore  # noqa: E402

    try:
        restore(load_artifact(ARTIFACT), load_bundled_policy())
    except Exception as exc:  # noqa: BLE001
        fail(f"산출물과 코드가 어긋난다: {exc}. tools/export_artifact.py를 다시 돌려야 한다")
    ok("산출물과 코드가 맞물린다")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        host_dir = work / "host"
        host_dir.mkdir()

        # 3. 호스트가 폴백 없이 도는가
        from router.cli import main as router_main  # noqa: E402
        import contextlib
        import io

        for tier in TIERS:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = router_main(["--input", str(DEV_INPUT), "--tier", tier,
                                    "--output", str(host_dir / f"{tier}.json")])
            if code != 0 or "폴백" in err.getvalue():
                fail(f"호스트 {tier}가 폴백을 썼다")
        ok("호스트 실행 3등급 폴백 없음")

        # 4·5. 컨테이너 실행과 호스트 대조
        cont_dir = work / "container"
        for tier in TIERS:
            log = run_container(DEV_INPUT.parent, DEV_INPUT.name, cont_dir, tier)
            if "폴백" in log or "fallback" in log:
                fail(f"컨테이너 {tier}가 폴백을 썼다")
        ok("컨테이너 실행 3등급 폴백 없음")

        for tier in TIERS:
            host, cont = decisions(host_dir / f"{tier}.json"), decisions(cont_dir / f"{tier}.json")
            diff = sum(1 for k in host if host[k] != cont.get(k))
            if diff:
                fail(f"{tier}: 호스트와 컨테이너 선택이 {diff}개 다르다")
        ok("호스트와 컨테이너 선택 완전 일치")

        # 6. 감사 재실행
        original = json.loads(DEV_INPUT.read_text(encoding="utf-8"))
        episodes = list(original["episodes"])
        random.Random(20260812).shuffle(episodes)
        mapping, shuffled = {}, []
        for i, episode in enumerate(episodes):
            copy = dict(episode)
            new_id = f"audit-{i:06d}"
            mapping[new_id] = episode["episode_id"]
            copy["episode_id"] = new_id
            shuffled.append(copy)
        audit_dir = work / "auditin"
        audit_dir.mkdir()
        (audit_dir / "inputs.json").write_text(
            json.dumps({**original, "episodes": shuffled}, ensure_ascii=False),
            encoding="utf-8",
        )
        audit_out = work / "auditout"
        for tier in TIERS:
            run_container(audit_dir, "inputs.json", audit_out, tier)
            base = decisions(cont_dir / f"{tier}.json")
            moved = {mapping[k]: v for k, v in decisions(audit_out / f"{tier}.json").items()}
            if base != moved:
                diff = sum(1 for k in base if base[k] != moved.get(k))
                fail(f"{tier}: ID·순서 감사에서 {diff}개 달라졌다")
        ok("컨테이너 ID·순서 감사 통과")

        # 7. 공식 채점기
        score_dir = work / "score"
        score_dir.mkdir()
        for tier in TIERS:
            shutil.copy(cont_dir / f"{tier}.json", score_dir / f"{tier}.json")
        report = work / "report.json"
        graded = subprocess.run(
            [sys.executable, "-m", "ossp_router.cli", "self-check",
             "--input", str(DEV_INPUT),
             "--outcomes", str(ROOT / "data" / "dev" / "outcomes.json"),
             "--submissions", str(score_dir), "--report", str(report)],
            cwd=ROOT, capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
        )
        if graded.returncode != 0:
            fail(f"공식 채점기 거부: {graded.stderr[:300]}")
        data = json.loads(report.read_text(encoding="utf-8"))
        for tier in TIERS:
            if not data["tiers"][tier]["budget_passed"]:
                fail(f"{tier} 예산 초과")
        ok("공식 채점기 통과, 세 등급 예산 통과")

        print("\n등급별 결과")
        for tier in TIERS:
            row = data["tiers"][tier]
            print(f"  {tier:9s} {row['quality_score'][:6]:>7s}  "
                  f"비율 {row['budget_ratio'][:5]:>6s}/{row['budget_multiplier']}")
        print(f"  {'가중 최종':9s} {data['final_score'][:6]:>7s}")

    print("\n제출 스택 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
