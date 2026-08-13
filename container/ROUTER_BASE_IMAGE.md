<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 제출 이미지의 기반 이미지와 포함 파일

`container/router.Dockerfile`이 만드는 제출 이미지의 출처·버전·라이선스를
기록한다. 저장소가 제공하는 예제(`container/Dockerfile`)와 격리 조건은 같고
기반 이미지만 다르다.

## 왜 예제 기반 이미지를 쓰지 않는가

예제는 `python:3.11.15-alpine3.23`을 고정한다. 우리 라우터는 numpy를 쓰는데
musl 환경에는 미리 만들어진 wheel이 없어 소스 빌드가 필요하다. Debian slim에는
manylinux wheel이 있어 빌드 도구 없이 설치된다. `docs/RUNTIME.md`는 예제 기반
이미지를 강제하지 않는다.

## 기반 이미지

- 참조: `python:3.11.15-slim-bookworm`
- 고정 다이제스트:
  `sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3`
- 선택 플랫폼: `linux/arm64`
- 출처: [Docker Official Images의 Python 항목](https://github.com/docker-library/official-images/blob/master/library/python)
- 구성: Debian 12.15 (bookworm), Python 3.11.15

이 저장소의 Apache-2.0 라이선스는 기반 이미지 안의 Python, Debian과 개별
패키지를 재라이선스하지 않는다. 이미지를 배포할 때는 Python 저작권·라이선스와
Debian 패키지 메타데이터의 고지 조건을 그대로 보존한다.

다이제스트 고정은 자동 보안 갱신을 막는다. 공개 배포 전에는 해당 다이제스트의
운영체제·Python 패키지 취약점을 다시 검사한다. 보안 갱신으로 기반 이미지를
바꾸면 새 이미지의 출처·라이선스·취약점을 함께 검증하고 제출 이미지
다이제스트를 새로 기록한다.

## 런타임 의존성

| 이름 | 버전 | 라이선스 | 출처 | 용도 |
| --- | --- | --- | --- | --- |
| numpy | 2.2.6 | BSD-3-Clause | [PyPI](https://pypi.org/project/numpy/2.2.6/) | 특징 배열과 배분 계산 |

`container/router-requirements.txt`에 wheel의 SHA-256을 고정하고
`pip install --require-hashes --no-deps`로 설치한다. 실행 중 다운로드는 없다.
`BSD-3-Clause`는 과제 규칙의 허용 라이선스 목록에 있다.

빌드가 끝난 뒤 `pip`, `setuptools`, `wheel`을 제거해 최종 이미지에서 뺀다.

우리가 설치하는 것은 numpy뿐이지만 최종 이미지의 site-packages에는
`packaging` 26.3이 하나 더 남는다. 기반 이미지가 넣은 것이고 위 제거 대상
이름에 없어 살아남는다. SBOM을 만들면서 확인했다.

| 이름 | 버전 | 라이선스 | 출처 | 라우터가 적재하는가 |
| --- | --- | --- | --- | --- |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause | 기반 이미지 | **아니오** |

두 라이선스 모두 과제 규칙의 허용 목록에 있다. 라우터 모듈을 전부 import한
뒤 `sys.modules`를 확인해 `packaging`이 적재되지 않음을 확인했으므로
"라우터 애플리케이션에 직접 결합하는" 구성요소가 아니다.

## 이미지에 싣는 참가자 파일

| 경로 | 내용 | 라이선스 |
| --- | --- | --- |
| `/opt/router/router/` | 라우터 구현 | Apache-2.0 (chjnett) |
| `/opt/router/ossp_router/` | 저장소가 제공하는 프로토콜·정책 | Apache-2.0 (SK Telecom) |
| `/opt/router/router/resources/artifact.v1.json` | 학습 산출물 | Apache-2.0 (chjnett) |
| `/opt/router/entrypoint.py` | 진입점 | Apache-2.0 (chjnett) |

학습 산출물에는 **계수만** 들어간다. 프롬프트 원문, 문항 ID, 문항별 선택은
넣지 않으며 `tests/test_router_cli.py`가 `state` 안에 문자열 값이 하나도 없는지
검사한다. 생성 절차와 입력 파일 해시는 산출물의 `provenance`에 기록한다.

산출물은 공개 Train 1,760문항으로만 적합한다. Dev는 보정 전용이라 섞지 않는다.
원천 자료의 출처와 라이선스는 저장소의 `THIRD_PARTY_NOTICES.md`를 따른다.

## 이미지 크기

공식 한도는 압축 계층 합계 1 GiB, 풀린 rootfs 겉보기 크기 2 GiB다.

`router-measure-image`는 레지스트리 다이제스트를 요구한다. 2026-08-13에
로컬 레지스트리로 push해 **공식 도구로 직접 측정**했다. 아래는 근사가 아니라
`operator-image-size-evidence` 산출물의 값이다.

| 지표 | 측정값 | 한도 | 사용률 |
| --- | ---: | ---: | ---: |
| 압축 계층 합계 (`oci-manifest-layer-descriptors-v1`) | 63.4 MiB | 1024 MiB | 6.2% |
| rootfs 겉보기 크기 (`docker-export-tar-apparent-size-v1`) | 195.8 MiB | 2048 MiB | 9.6% |

같은 이미지를 `tools/check_runtime.py`로 공식 자원 한도(2코어, 2 GiB, 네트워크
없음, 읽기 전용 루트)에서 돌려 Train+Dev 2,640문항이 등급당 2.1초에 끝나는
것을 확인했다. 한도는 등급당 90초다.

측정 절차는 [`plan/RELEASE.md`](../plan/RELEASE.md)에 있다. push한 이미지와
OCI 레이아웃을 **같은 미디어타입·압축으로** 내보내지 않으면 매니페스트
다이제스트가 갈려 측정이 거부된다.

## SBOM

[`container/sbom.cdx.json`](sbom.cdx.json) — CycloneDX 1.5. 구성요소 108개
(Debian 105, Python 2, 참가자 1).

```console
PYTHONPATH=src python3 tools/generate_sbom.py \
  --image <REGISTRY>/ossp-router@sha256:<64자리>
```

`syft`나 `trivy`를 쓰지 않고 **이미지 안의** dpkg 데이터베이스와 Python
dist-info를 직접 읽는다. 출처가 이미지 자신이라 "이 다이제스트에 실제로 든
것"과 어긋날 수 없다. 바깥에서 만든 목록은 이미지와 갈릴 수 있다.

기반 OS 패키지의 라이선스 근거는 이미지 안 `/usr/share/doc/<패키지>/copyright`에
있다. slim 이미지가 이 파일들을 지우는 경우가 있어 확인했고, 105개 패키지
전부 보존되어 있다. SBOM은 각 구성요소에 그 경로를 `ossp:license-file`로
기록하고, DEP-5 형식인 88개는 라이선스 식별자까지 읽어 넣는다.

Debian 기반 이미지에는 GPL 계열 시스템 패키지가 들어간다. 과제 규칙은 기반
운영체제와 언어 런타임의 표준 구성요소를 허용하며, 금지 대상은 **라우터
애플리케이션에 직접 결합하는** 목록 밖 copyleft다. 라우터가 결합하는 것은
numpy(BSD-3-Clause) 하나뿐이다.

## 재현

```console
docker build --platform linux/arm64 --provenance=false --sbom=false \
  --file container/router.Dockerfile --tag ossp-router:local .

PYTHONPATH=src python3 tools/verify_submission_stack.py
```

`--provenance=false --sbom=false`가 필요하다. 이것을 빼면 buildx가 attestation
매니페스트를 붙여 `tools/check_runtime.py`가 고정한 이미지 ID로 실행하지 못한다.
