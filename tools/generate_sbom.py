# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

# 이 파일은 라이선스 메타데이터를 다루는 도구라 본문에 `License:` 같은
# 토큰이 들어간다. reuse가 그것을 이 파일 자신의 헤더로 오인해 위의 두 줄을
# 놓친다. 아래부터 파일 끝까지는 헤더 탐지에서 제외한다.
# REUSE-IgnoreStart

"""제출 이미지의 SBOM을 만든다 (CycloneDX 1.5 JSON).

결과보고서 요구 항목이다. syft·trivy 같은 도구를 쓰지 않고 이미지 안의
dpkg 데이터베이스와 Python dist-info를 그대로 읽는다. 이유는 두 가지다.

* 그 도구들이 이 장비에 없고, 마감 직전에 새 도구를 들이면 결과를 검증할
  시간이 없다.
* 출처가 이미지 자신이라 "이 다이제스트에 실제로 든 것"과 어긋날 수 없다.
  바깥에서 만든 목록은 이미지와 갈릴 수 있다.

    PYTHONPATH=src python3 tools/generate_sbom.py \
      --image <REGISTRY>/ossp-router@sha256:... \
      --output container/sbom.cdx.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]

# 이미지 안에서 실행한다. 바깥 추정이 아니라 이미지가 스스로 말하게 한다.
_COLLECT = r"""
import json, os, re

def dpkg():
    path = "/var/lib/dpkg/status"
    if not os.path.exists(path):
        return []
    out, block = [], {}
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                if block.get("Package"):
                    out.append(block)
                block = {}
                continue
            if line.startswith((" ", "\t")):
                continue
            key, _, value = line.partition(":")
            block[key.strip()] = value.strip()
    if block.get("Package"):
        out.append(block)
    packages = []
    for b in out:
        if not b.get("Status", "").endswith("installed"):
            continue
        name = b["Package"]
        # 라이선스 근거는 이미지 안 /usr/share/doc/<pkg>/copyright에 실려
        # 있다. slim 이미지가 지우기도 하는데 이 기반 이미지는 105개 전부
        # 보존한다. DEP-5 형식이면 License: 줄을 그대로 읽는다.
        path = "/usr/share/doc/%s/copyright" % name
        licenses, has_copyright = [], os.path.exists(path)
        if has_copyright:
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    found = re.match(r"^License:[ \t]*(\S.*)$", line.rstrip("\n"))
                    if found:
                        value = found.group(1).strip()
                        if value not in licenses and len(value) < 60:
                            licenses.append(value)
        packages.append({
            "name": name,
            "version": b.get("Version", ""),
            "arch": b.get("Architecture", ""),
            "source": b.get("Source", "").split(" ")[0] or name,
            "licenses": licenses,
            "copyright_file": path if has_copyright else "",
            "kind": "deb",
        })
    return packages

def python_packages():
    out = []
    seen = set()
    for root in ("/usr/local/lib", "/usr/lib", "/opt/router"):
        for cur, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for d in list(dirs):
                if not d.endswith((".dist-info", ".egg-info")):
                    continue
                meta = os.path.join(cur, d, "METADATA")
                if not os.path.exists(meta):
                    meta = os.path.join(cur, d, "PKG-INFO")
                if not os.path.exists(meta):
                    continue
                # License 필드에 라이선스 전문이 통째로 들어 있기도 하다.
                # numpy가 그렇고, 그 전문 안의 빈 줄을 헤더 끝으로 오인하면
                # 뒤에 오는 Classifier까지 못 간다. 빈 줄로 끊지 말고 헤더처럼
                # 생긴 줄만 골라낸다. 본문에 우연히 걸리는 줄이 있어도 앞선
                # 값이 이기므로 해가 없다.
                header = re.compile(r"^([A-Za-z][A-Za-z0-9-]*):[ \t]*(.*)$")
                fields, classifiers = {}, []
                with open(meta, encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        found = header.match(line.rstrip("\n"))
                        if not found:
                            continue
                        key, value = found.group(1).lower(), found.group(2).strip()
                        if key == "classifier" and value.startswith("License ::"):
                            classifiers.append(value.rsplit("::", 1)[-1].strip())
                        else:
                            fields.setdefault(key, value)
                name = fields.get("name")
                version = fields.get("version", "")
                if not name or (name, version) in seen:
                    continue
                seen.add((name, version))
                declared = fields.get("license-expression", "")
                if not declared and classifiers:
                    declared = " OR ".join(classifiers)
                out.append({
                    "name": name,
                    "version": version,
                    "license": declared,
                    "license_source": (
                        "License-Expression" if fields.get("license-expression")
                        else ("Classifier" if classifiers else "")
                    ),
                    "homepage": fields.get("home-page", ""),
                    "kind": "pypi",
                })
    return out

import sys
print(json.dumps({
    "deb": dpkg(),
    "pypi": python_packages(),
    "python": sys.version.split()[0],
}))
"""


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"명령이 실패했다: {' '.join(command[:5])}\n{result.stderr[:500]}")
    return result.stdout


def _purl(entry: dict[str, Any]) -> str:
    name = quote(entry["name"], safe="")
    version = quote(entry.get("version", ""), safe="")
    if entry["kind"] == "deb":
        arch = entry.get("arch", "")
        suffix = f"?arch={quote(arch, safe='')}" if arch else ""
        return f"pkg:deb/debian/{name}@{version}{suffix}"
    return f"pkg:pypi/{name}@{version}"


def build_document(image: str, collected: dict[str, Any], base_image: str) -> dict[str, Any]:
    components: list[dict[str, Any]] = []

    for entry in sorted(collected["deb"], key=lambda e: (e["name"], e["version"])):
        component: dict[str, Any] = {
            "type": "library",
            "name": entry["name"],
            "version": entry["version"],
            "purl": _purl(entry),
            "scope": "required",
            "properties": [
                {"name": "ossp:origin", "value": "base-image-debian"},
                {"name": "ossp:source-package", "value": entry["source"]},
            ],
        }
        if entry.get("copyright_file"):
            component["properties"].append(
                {"name": "ossp:license-file", "value": entry["copyright_file"]}
            )
        if entry.get("licenses"):
            component["licenses"] = [
                {"license": {"name": name}} for name in entry["licenses"]
            ]
        components.append(component)

    for entry in sorted(collected["pypi"], key=lambda e: (e["name"], e["version"])):
        component: dict[str, Any] = {
            "type": "library",
            "name": entry["name"],
            "version": entry["version"],
            "purl": _purl(entry),
            "scope": "required",
            "properties": [{"name": "ossp:origin", "value": "pip-require-hashes"}],
        }
        declared = entry.get("license", "").strip()
        if declared and len(declared) < 100:
            component["licenses"] = [{"license": {"name": declared}}]
            component["properties"].append(
                {"name": "ossp:license-source", "value": entry.get("license_source", "")}
            )
        if entry.get("homepage"):
            component["externalReferences"] = [
                {"type": "website", "url": entry["homepage"]}
            ]
        components.append(component)

    components.append(
        {
            "type": "application",
            "name": "ossp-router",
            "version": "1.0.0",
            "scope": "required",
            "licenses": [{"license": {"id": "Apache-2.0"}}],
            "properties": [
                {"name": "ossp:origin", "value": "participant-source"},
                {
                    "name": "ossp:note",
                    "value": "라우터 구현과 굳힌 계수. AI 모델은 포함하지 않는다",
                },
            ],
        }
    )

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "container",
                "name": image.split("@")[0],
                "version": image.split("@")[-1] if "@" in image else "unknown",
                "licenses": [{"license": {"id": "Apache-2.0"}}],
            },
            "properties": [
                {"name": "ossp:base-image", "value": base_image},
                {"name": "ossp:python", "value": collected["python"]},
                {"name": "ossp:platform", "value": "linux/arm64"},
                {
                    "name": "ossp:method",
                    "value": "이미지 안 dpkg status와 Python dist-info를 직접 읽었다",
                },
                {
                    "name": "ossp:ai-models",
                    "value": "해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음",
                },
                {"name": "ossp:network-at-runtime", "value": "none"},
            ],
        },
        "components": components,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="제출 이미지의 CycloneDX SBOM을 만든다.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "container" / "sbom.cdx.json")
    args = parser.parse_args(argv)

    base_image = ""
    dockerfile = (ROOT / "container" / "router.Dockerfile").read_text(encoding="utf-8")
    for line in dockerfile.splitlines():
        if line.startswith("FROM "):
            base_image = line[5:].strip()
            break

    collected = json.loads(
        _run(
            [
                "docker", "run", "--rm", "--network", "none",
                "--entrypoint", "python3", args.image, "-c", _COLLECT,
            ]
        )
    )
    document = build_document(args.image, collected, base_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    # JSON에는 주석을 넣을 수 없어 REUSE sidecar를 따로 쓴다.
    sidecar = args.output.with_name(args.output.name + ".license")
    sidecar.write_text(
        "SPDX-FileCopyrightText: Copyright 2026 chjnett\n"
        "SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    # REUSE-IgnoreEnd
    debs = sum(1 for c in document["components"] if c.get("purl", "").startswith("pkg:deb"))
    pypi = sum(1 for c in document["components"] if c.get("purl", "").startswith("pkg:pypi"))
    print(
        f"OK: {args.output} — 구성요소 {len(document['components'])}개 "
        f"(Debian {debs}, Python {pypi}, 참가자 1)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# REUSE-IgnoreEnd
