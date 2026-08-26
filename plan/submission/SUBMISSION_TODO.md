<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 제출 TODO — 마감 2026-08-27(목) 18:00 KST

공식 문서: `docs/SUBMISSION.md`(절차), `docs/ENFORCEMENT.md`(실패 분류),
`docs/RUNTIME.md`(컨테이너), `schemas/technical-submission.v1.schema.json`(JSON 형식).
자체 절차: `plan/RELEASE.md`. 확인 도구: `tools/validate_technical_submission.py`,
`tools/verify_release.py`, `tools/check_runtime.py`, `tools/verify_submission_stack.py`.

## 핵심 실패 분류 (ENFORCEMENT.md — 반드시 알아야 할 것)

| 상황 | 결과 |
| --- | --- |
| 예산 초과 | 그 등급만 0점 |
| 실행·형식 오류 (3회 모두) | 그 등급 0점 |
| **이미지 사전검증 실패 (크기·VOLUME)** | **실행도 안 하고 접수 거부 — 재시도 없음** |
| 금지 전략·격리 우회 | 전체 실격 |
| repo 미공개 / 커밋-이미지 불일치 | 제출 무효 |

---

## A. 챔피언 확정 (지금 — 8/19)

- [x] 챔피언 T38 (`t38-prem-q068-urb120`) 선정, Docker 0.683892
- [x] **T43 파레토 우세로 승격** (`t43-bal-h112`, 8/26) — 공식 채점 가중 **0.6844**,
      게이트 26/12,000 C4 통과(T38과 동일), 이미지 `970e504f`·커밋 `35a8398`
- [ ] 최종 재검증: 4/6/8-fold OOF, LOFO, 스트레스 26/12,000
- [ ] **8/19 24:00: 새 트랙 착수 금지** (이후 점수·임계값 변경 없음)

## B. 제출 산출물 (8/20–8/24) — 코드 커밋 전에 마무리

- [ ] 아티팩트 최종 export (`tools/export_artifact.py --config t38...`) — **이미지 동결 전**
- [ ] **마지막 `--rebuild`**: `tools/verify_submission_stack.py --rebuild`
      (7개 검사 + 세 등급 예산 통과, 가중 > all-light 0.619)
- [ ] 검증 명령 3종 통과 (E9):
      `python3 -m unittest discover -s tests -p 'test_*.py'` / `ruff check .` / `reuse lint`
- [ ] **코드 커밋 → push** ← 이 커밋 SHA가 `commit_sha`가 됨
- [ ] **이 시점 이후 `--rebuild` 금지** (산출물 재생성 금지 — provenance 어긋남)

## C. 이미지 빌드·검증·동결 (8/25)

- [ ] `SOURCE_MANIFEST_SHA256` 계산 (`tools/benchmark_runtime.py --print-source-manifest-sha256`)
- [ ] `docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
       --build-arg SOURCE_MANIFEST_SHA256=$MAN -f container/router.Dockerfile`
- [ ] 공개 registry push (GHCR/Docker Hub, **로그아웃 상태 pull 가능해야 함**)
- [ ] 전체 digest 확보: `docker buildx imagetools inspect <registry>/ossp-router:submission`
- [ ] **공식 크기 측정** (한도: 압축 1GiB / rootfs 2GiB): `router-measure-image`
- [ ] **런타임 검증** (한도: 90s / 2GiB / 32pids): `tools/check_runtime.py`
- [ ] `VOLUME` 선언 없음 확인 (verify_release가 검사)
- [ ] **8/25 24:00 이미지 동결** — 이후 라우터 코드 수정 금지

## D. 기술 제출 JSON (8/26) — 별도 커밋

- [ ] 저장소 루트에 `submission-ossp-skt.json` 생성 (6필드만, 형식은 schema)
      ```json
      {"schema_version":1,
       "challenge_id":"ossp-2026-llm-router-challenge",
       "repository_url":"https://github.com/chjnett/ossp-2026-llm-router-challenge",
       "commit_sha":"<B단계 코드 커밋 40자리>",
       "image_digest":"<registry>/ossp-router@sha256:<64자리>",
       "primary_license":"Apache-2.0"}
      ```
- [ ] `tools/validate_technical_submission.py` 통과
- [ ] `tools/verify_release.py --evidence build/measure/evidence.json` **9항목 전부** 통과
      (트리 깨끗 / 스키마 / URL=fork / 커밋 공개 / digest 존재 / VOLUME 없음 /
      이미지가 커밋 코드 포함 / 아티팩트 비낡음 / 크기 한도)
- [ ] **JSON만 별도 커밋 + push** (코드 섞지 말 것)

## E. repo 공개 확인 (8/25–8/26)

- [ ] **제출 시점에 반드시 public** (개발 중 private 허용 — `SUBMISSION.md` "최종 제출
      시점부터 평가가 끝날 때까지 별도 권한 없이 열 수 있어야")
- [ ] `curl -s https://api.github.com/repos/chjnett/ossp-2026-llm-router-challenge`
      → 인증 없이 200
- [ ] 제출 커밋 스냅샷 URL = `tree/<40자리>` 형식 (브랜치 URL 아님)

## F. 결과보고서 (8/26–8/27) — `plan/REPORT_FORM.md` 참고

- [ ] `REPORT_FORM.md` 내용 → 공식 양식(HWP/DOC)에 채우기, 회색 문구 삭제
- [ ] 프로젝트 등록 URL = **JSON 담은 커밋**의 `tree/<SHA>` 스냅샷
- [ ] 시연영상 유튜브 URL (2–3분: docker run → self-check → verify_release)
- [ ] SBOM = `container/sbom.cdx.json` (108개)
- [ ] 붙임2: AI 모델 "해당 없음" + 상용 AI 도구 범위 (Claude·Codex)
- [ ] 5페이지 이내, 맑은고딕 10pt
- [ ] 원본 1부 + PDF 1부, 파일명 `2026 오픈소스 개발자대회 결과보고서_접수번호(팀명)`

## G. 제출 (8/27 18:00 KST 전)

- [ ] osscontest.kr 업로드 (마감 전 재업로드 가능, **마지막 접수분**이 심사)
- [ ] 18:00 이후 모든 제출·수정 차단

---

## 사용자 몫 (⬜, 지금 준비 시작)

- [ ] 공개 컨테이너 registry 계정 (GHCR 권장 — GitHub과 연동)
- [ ] 접수번호·팀명·참가부문 확인
- [ ] 데모 영상 촬영 (2–3분)
- [ ] 공식 결과보고서 양식 다운로드 (osscontest.kr)

## 최종 제출 전 5분 체크 (SUBMISSION.md 체크리스트)

1. 공개 fork에서 제출 커밋을 별도 권한 없이 열 수 있다
2. 심사에 필요한 전체 소스가 제출 커밋에 있다
3. 같은 커밋에서 빌드한 linux/arm64 이미지가 공개 레지스트리에 있다
4. 이미지 참조가 `@sha256:` 전체 digest다 (태그 아님)
5. Train+Dev 세 등급 실행시간·형식 확인했다
6. 파일 라이선스 근거가 공개되어 있다
7. `submission-ossp-skt.json`이 스키마 통과 + 최종 커밋에 있다
8. 보고서 URL이 JSON 포함 정확한 커밋 스냅샷을 가리킨다
