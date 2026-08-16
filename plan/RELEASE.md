<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 제출 실행 절차

2026-08-13에 로컬 레지스트리로 전 과정을 리허설하고, 2026-08-16 현재 제출
챔피언으로 다시 검증한 문서다.
마감 당일에 다시 알아내지 않으려고 쓴다. **위에서 아래로 그대로 실행한다.**

마감은 2026-08-27 18:00 KST. 이미지 동결은 8/25 24:00 (RULES F2).

---

## 리허설에서 걸린 것 (같은 실수를 반복하지 않기 위해)

| 증상 | 원인 | 대응 |
| --- | --- | --- |
| `router-measure-image`가 "제출 digest를 OCI layout index에서 찾을 수 없다" | push한 이미지와 OCI 레이아웃을 **따로** 빌드했다. Docker 미디어타입과 OCI 미디어타입은 매니페스트 다이제스트가 다르다 | 두 내보내기 모두 `oci-mediatypes=true, compression=gzip` |
| 이미지 라벨이 `unbound` | Dockerfile에 자리만 있고 build arg를 안 넘겼다. 라벨 키도 상류가 읽는 것과 달랐다 | `--build-arg SOURCE_MANIFEST_SHA256=...` 필수 |
| 관문이 push한 커밋을 "origin에 없다"고 판정 | `remote.origin.fetch`가 `main`으로 제한돼 있어 다른 브랜치의 원격 추적 ref가 안 생긴다 | `git ls-remote`로 원격에 직접 묻는다 |
| 관문이 정상 이미지를 실패로 판정 | 이미지와 커밋의 **동일성**을 요구했다. `.dockerignore`가 학습 코드(`cache`·`harness`·`stress`)를 일부러 뺀다 | 부분집합으로 대조 |
| 소스 매니페스트가 이미지 라벨과 불일치 | `verify_submission_stack.py --rebuild`가 산출물을 다시 구워 `provenance`의 커밋·시각이 바뀌었다. 계수는 동일하다 | **산출물 재생성은 코드 커밋 전에 끝낸다.** 커밋 뒤에는 `--rebuild`를 돌리지 않는다 |

비트 단위 재현 빌드는 **성립하지 않는다.** pip 설치 시각이 레이어에 들어가
같은 커밋에서 두 번 빌드해도 다이제스트가 다르다. 규칙이 요구하지 않으므로
내용 대조(`tools/verify_release.py`)로 대신한다.

### 이 이미지는 arm64에서만 구울 수 있다

`container/router-requirements.txt`가 numpy wheel의 SHA-256을 고정하는데 그
wheel은 **aarch64용**이다. amd64에서는 pip가 x86_64 wheel을 받아
`--require-hashes`가 막는다. 기반 이미지도 arm64 다이제스트로 고정돼 있다.

QEMU 에뮬레이션으로 우회하지 말고 arm64 장비에서 굽는다. Apple Silicon이면
그대로 되고, CI는 `ubuntu-24.04-arm` 러너를 쓴다(공개 저장소 무료).

### Linux에서만 드러나는 것

컨테이너는 UID 65532로 돈다. Linux의 bind mount는 호스트의 소유권과 권한을
그대로 보여주므로 **0700 디렉터리를 마운트하면 컨테이너가 읽지 못한다.**
macOS Docker Desktop은 소유권을 느슨하게 매핑해서 이 문제가 로컬에서는
보이지 않는다. 검증 도구가 CI(Linux)에서 처음 깨졌다.

로컬에서만 확인하고 넘어가면 안 되는 이유다. **평가 환경은 Linux arm64다.**

---

## 0. 사전 조건

```bash
docker version && git status --porcelain
```

작업 트리가 **비어 있어야** 한다. `linux/arm64` 빌드가 가능해야 한다
(Apple Silicon이면 그대로 된다).

공개 레지스트리 계정이 있어야 한다. Docker Hub, GHCR 어느 쪽이든 되지만
**로그아웃 상태에서 pull이 되어야** 한다.

### 브랜치

개발은 `router-dev`에서 했다. 규칙상 브랜치 이름은 자유이고 제출 대상을
고정하는 것은 커밋 SHA다. 다만 **제출 커밋은 `main`에 올려 둔다.** 작업
브랜치는 지워질 수 있고, 그러면 심사 기간 중에 스냅샷 URL이 죽는다.

```bash
git checkout main && git merge --ff-only router-dev && git push origin main
```

### fork 공개 확인 (2026-08-13 확인)

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  https://api.github.com/repos/chjnett/ossp-2026-llm-router-challenge
```

인증 없이 `200`이어야 한다. `private=False`를 확인했다. 제출 시점에 다시
확인한다 — 심사 기간 내내 열려 있어야 한다 (RULES F4).

## 1. 챔피언 확정과 산출물 굽기 — 코드 커밋 **전에**

```bash
PYTHONPATH=src python3 tools/export_artifact.py \
  --config experiments/configs/t30-headroom-b1075-p100.json
```

이미지 동결 전에 챔피언이 바뀌면 위 설정 경로도 반드시 함께 바꾼다.
`experiments/champion.json`의 ID와 아티팩트 `config.id`가 다르면 다음 단계의
`verify_submission_stack.py`가 실패해야 정상이다.

`provenance.commit`은 굽는 시점의 HEAD를 적는다. 그래서 산출물이 담기는
커밋보다 항상 하나 앞선다. 정상이다.

## 2. 전 스택 검증 — 여기까지가 마지막 `--rebuild`

```bash
PYTHONPATH=src python3 tools/verify_submission_stack.py --rebuild
```

7개 검사가 전부 통과하고 세 등급 예산이 통과해야 한다. 가중 최종 점수가
all-light(0.619)보다 낮으면 제출하지 않는다 (RULES F1).

## 3. 코드 커밋과 push

```bash
git add -A && git commit && git push origin HEAD
```

이 커밋이 `commit_sha`가 된다. **이 시점 이후 `--rebuild` 금지.**

## 4. 이미지 빌드와 push

레지스트리는 공개여야 한다. `<REGISTRY>`를 실제 값으로 바꾼다.

```bash
MAN="$(PYTHONPATH=src python3 tools/benchmark_runtime.py --print-source-manifest-sha256)"
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --build-arg "SOURCE_MANIFEST_SHA256=$MAN" \
  --file container/router.Dockerfile \
  --output "type=registry,name=<REGISTRY>/ossp-router:submission,oci-mediatypes=true,compression=gzip" .
```

`--provenance=false --sbom=false`가 빠지면 buildx가 attestation 매니페스트를
붙여 고정 이미지 ID로 실행하지 못한다.

같은 조건으로 OCI 레이아웃도 뽑는다. **미디어타입과 압축이 위와 같아야
다이제스트가 일치한다.**

```bash
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --build-arg "SOURCE_MANIFEST_SHA256=$MAN" \
  --file container/router.Dockerfile \
  --output "type=oci,dest=build/oci,tar=false,compression=gzip" .
```

다이제스트를 확보하고 로컬로 당긴다.

```bash
docker buildx imagetools inspect <REGISTRY>/ossp-router:submission
docker pull --platform linux/arm64 <REGISTRY>/ossp-router@sha256:<64자리>
```

## 5. 공식 측정

```bash
mkdir -p build/measure && chmod 700 build/measure
PYTHONPATH=src router-measure-image \
  --oci-layout build/oci \
  --image <REGISTRY>/ossp-router@sha256:<64자리> \
  --output build/measure/evidence.json
```

리허설 측정값 (한도 대비):

| 지표 | 측정 | 한도 | 사용률 |
| --- | ---: | ---: | ---: |
| 압축 계층 합계 | 63.4 MiB | 1024 MiB | 6.2% |
| rootfs 겉보기 | 195.8 MiB | 2048 MiB | 9.6% |

## 6. 공식 자원 한도로 실행 확인

```bash
PYTHONPATH=src python3 tools/check_runtime.py \
  --image <REGISTRY>/ossp-router@sha256:<64자리> \
  --report build/runtime-check-report.json
```

Train+Dev 2,640문항으로 세 등급이 각각 90초 안에 끝나야 한다. 2026-08-15
현재 T17 이미지의 실행은 Fast 8.116초, Balanced 7.875초, Premium
7.903초였다.

## 7. 기술 제출 JSON — **별도 커밋**

저장소 루트에 `submission-ossp-skt.json`. 여섯 필드만 허용한다.

```json
{
  "schema_version": 1,
  "challenge_id": "ossp-2026-llm-router-challenge",
  "repository_url": "https://github.com/chjnett/ossp-2026-llm-router-challenge",
  "commit_sha": "<3단계 커밋의 40자리>",
  "image_digest": "<REGISTRY>/ossp-router@sha256:<64자리>",
  "primary_license": "Apache-2.0"
}
```

이 커밋에는 **JSON만** 넣는다. 코드가 섞이면 `commit_sha`가 가리키는 커밋과
제출 스냅샷의 코드가 갈린다.

## 8. 관문 통과 — 제출 전 마지막 확인

```bash
PYTHONPATH=src python3 tools/validate_technical_submission.py
PYTHONPATH=src python3 tools/verify_release.py --evidence build/measure/evidence.json
```

공식 검증기는 **형식만** 본다. 커밋 SHA는 맞고 이미지는 사흘 전 코드로 구운
제출이 그대로 통과한다. `verify_release.py`가 9개 항목으로 내용을 본다.
하나라도 실패하면 제출하지 않는다.

```bash
git add submission-ossp-skt.json && git commit && git push origin HEAD
```

## 9. 결과보고서

`프로젝트 등록 URL`에 **JSON을 담은 8단계 커밋**의 스냅샷을 적는다.

```
https://github.com/chjnett/ossp-2026-llm-router-challenge/tree/<40자리>
```

브랜치 URL이 아니라 전체 커밋 SHA여야 한다. 요구 항목은
[`plan/REPORT.md`](REPORT.md)에 정리한다.

## 10. 접수

[osscontest.kr](https://osscontest.kr/)에 원본 파일 1개와 PDF 1개.
본문 5쪽 이내, 첫 쪽 안내 문구 삭제, 파일명
`2026 오픈소스 개발자대회 결과보고서_접수번호(팀명)`.

마감 전에는 다시 올릴 수 있고 **마지막 접수분**을 심사한다.

---

## 제출 후

저장소는 심사 종료까지, 수상 시 5년간 공개 유지 (RULES F4).
상류로 PR을 보낼 필요는 없다. 수상작에 한해 심사 후 별도로 요청받을 수 있다.
