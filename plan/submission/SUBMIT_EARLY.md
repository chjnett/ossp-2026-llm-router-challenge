<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# "조기 제출 + 지속 수정" 전략 설계

마감 2026-08-27(목) 18:00 KST. 목표: **유효한 제출을 먼저 확보한 뒤, 마감까지
계속 개선해 재제출**한다.

## 1. 규칙 근거 (왜 가능한가)

`docs/SUBMISSION.md`:

> "마감 전에는 결과보고서를 복수로 제출하거나 자유롭게 다시 업로드할 수 있으며
> **마지막으로 접수된 파일을 심사**합니다. 8월 27일 18:00 이후에는 새 제출과
> 수정이 모두 차단됩니다."

제출 대상 = 결과보고서에 적은 `프로젝트 등록 URL`(= `submission-ossp-skt.json`을
담은 커밋 스냅샷). 그 JSON이 **코드 커밋 SHA + 이미지 digest**를 가리킨다.

→ **보고서를 재업로드할 때마다 그 URL이 가리키는 커밋·이미지가 제출 대상이 된다.**
즉 라우터를 개선 → 새 커밋 → 새 이미지 → JSON 갱신 → 보고서 URL 갱신 → 재업로드,
이 사이클을 마감까지 반복할 수 있다.

### 지켜야 할 공식 제약

| 제약 | 근거 |
| --- | --- |
| 이미지는 **항상 그 커밋에서 빌드**한 digest여야 | `CHALLENGE_RULES` "제출한 커밋에서 재현 가능하게 빌드" |
| repo는 **첫 제출부터 평가 끝까지 공개** | `SUBMISSION.md` "최종 제출 시점부터 별도 권한 없이 열 수 있어야" |
| `submission-ossp-skt.json`은 **JSON만 담은 별도 커밋** | `SUBMISSION.md` "JSON만 추가한 별도 커밋" |
| 마감 후(18:00) 새 제출·수정 차단 | `SUBMISSION.md` |
| 등급별 실행·형식 오류는 최대 3회, 모두 실패 시 등급 0점 | `ENFORCEMENT.md` |
| 이미지 사전검증(크기·VOLUME) 실패는 **재시도 없이 접수 거부** | `ENFORCEMENT.md` |

## 2. 전략 개요

```
8/17-8/19  [1차 제출]  유효한 제출 확보 (안전망)
8/19-8/25  [개선 루프]  라우터 개선 → 재제출 (원할 때마다)
8/25-8/27  [최종 확정]  마지막 검증 후 최종 보고서 업로드
```

**핵심 원칙: "재업로드는 검증된 것만"** — 마지막 보고서가 심사되므로, 깨진 버전을
업로드하면 그게 최종본이 된다. 개선분은 반드시 `verify_release.py` 통과 후에만
업로드한다.

## 3. 1차 제출 (8/17-8/19) — 안전망 확보

1. **이미지 push**: `ossp-router:submission`을 공개 registry에 push, `@sha256:` digest 확보
   (사용자 몫: GHCR 계정 — GitHub과 연동, 로그아웃 pull 가능 확인)
2. **기술 JSON**: 저장소 루트에 `submission-ossp-skt.json` (6필드) 생성
3. **검증**: `validate_technical_submission.py` + `verify_release.py --evidence` 9항목
4. **JSON만 별도 커밋 + push**
5. **보고서 작성**: `plan/REPORT_FORM.md` → HWP/DOC + PDF, URL = JSON 커밋 `tree/<SHA>`
6. **osscontest.kr 업로드** ← 여기서 "유효한 제출"이 고정됨

→ 이후 어떤 수정이든, 이 1차 제출이 실패해도 **마지막으로 유효했던 버전보다 나쁘지
않도록** 항상 검증 후 업로드.

## 4. 개선 루프 (8/19-8/25) — 재제출 사이클

각 개선(새 코드)마다 다음을 실행한다. **스크립트로 자동화해 실수를 없앤다.**

```bash
# 1) 아티팩트 export (산출물이 코드에 딸려 가므로 커밋 전)
PYTHONPATH=src python3 tools/export_artifact.py --config <새 챔피언 config>

# 2) 전 스택 검증 (마지막 --rebuild)
PYTHONPATH=src python3 tools/verify_submission_stack.py --rebuild

# 3) 코드 커밋 + push  (이 SHA가 commit_sha)
git add -A && git commit && git push origin main

# 4) 이미지 빌드 + push + digest
MAN=$(PYTHONPATH=src python3 tools/benchmark_runtime.py --print-source-manifest-sha256)
docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
  --build-arg SOURCE_MANIFEST_SHA256=$MAN -f container/router.Dockerfile \
  --output "type=registry,name=<REGISTRY>/ossp-router:submission,oci-mediatypes=true,compression=gzip" .
docker buildx imagetools inspect <REGISTRY>/ossp-router:submission   # digest

# 5) JSON 갱신 (commit_sha + image_digest) → 별도 커밋
# 6) verify_release.py 9항목 통과
# 7) 보고서 URL을 새 JSON 커밋의 tree/<SHA>로 바꿔 재업로드
```

**자동화 스크립트** (`tools/resubmit.sh` 제안)가 이 7단계를 한 번에 수행하고,
어느 단계든 실패하면 그 개선분은 "업로드 후보에서 제외"한다.

## 5. 최종 확정 (8/25-8/27)

- 8/25 이후: **새 개선 없이** 마지막 검증만 반복 (`verify_release.py` + Docker self-check)
- 마지막 보고서 업로드는 **8/27 17:30 이전** (업로드 여유 확보)
- 18:00 이후 수정 차단

## 6. 리스크 관리

| 리스크 | 완화 |
| --- | --- |
| 최종 버전이 깨짐 | 재업로드는 항상 검증 후. 1차 제출(안전망)은 그대로 두되, **마지막 업로드만 심사**되므로 마지막 직전 버전을 검증 |
| 이미지 사전검증 거부 | 매 빌드 후 `router-measure-image` 크기 + VOLUME 검사 |
| repo 실수로 private | 첫 제출 전 public 고정, curl 200 체크를 스크립트에 포함 |
| JSON 커밋에 코드 섞임 | JSON만 `git add`하고 별도 커밋 (스크립트가 강제) |
| 커밋-이미지 불일치 | `verify_release.py`의 "이미지가 커밋 코드 포함" 검사가 차단 |

## 7. 지금 바로 할 것

1. **사용자 몫**: GHCR 계정 확보 + 공식 양식 다운로드 + 접수번호 확인
2. **자동화**: `tools/resubmit.sh` 작성 (위 7단계)
3. **1차 보고서**: `REPORT_FORM.md` 채우고 HWP/DOC + PDF 준비
4. **1차 제출 실행** (8/19 목표)
