<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 제출 체크리스트 — 사용자 몫 (경로 포함)

마감: **2026-08-27(목) 18:00 KST** · 접수번호: **379**

이 폴더(`plan/submission/`)에 제출 관련 문서가 모두 있습니다.

## 0. 이미 완료·검증된 것 (참고만)

| 항목 | 값 |
| --- | --- |
| 챔피언 | `t38-prem-q068-urb120` — Docker 0.683892 |
| 이미지 | `ghcr.io/chjnett/ossp-router@sha256:13b166e5d3c37277b8631aa4f6ea4c7faae893302b15d91110ecc275affbcc6d` |
| 코드 커밋 | `4f811afd1ee414b352665d006c7b4985cd7593a4` |
| JSON 커밋 | `22178dd` (main, origin push됨) |
| 제출 관문 | `tools/verify_release.py` 9/9 통과 · `validate_technical_submission.py` 통과 |
| 전체 절차 | `plan/submission/SUBMISSION_TODO.md` |

## 1. 지금 해야 할 것 (사용자 몫)

- [ ] **공식 결과보고서 양식 다운로드** — osscontest.kr > 대회 소개 > 대회 개요 > [결과보고서 양식 다운로드]
      (HWP/DOCX 원본 + PDF 변환 파일 2개 제출)
- [ ] **접수 정보 확인** — 팀명, 참가부문(학생/일반), 팀 인원
- [ ] **데모 영상 촬영·업로드 (3분 이내, 유튜브)** — 스크립트: `plan/submission/DEMO_SCRIPT.md`

## 2. 결과보고서 작성 (5페이지 이내, 맑은고딕 10pt)

원고: `plan/submission/REPORT_FORM.md` — 모든 항목이 채워져 있음, ⬜만 채우면 됨

- [ ] ⬜ 팀명 · 참가부문 기재
- [ ] ⬜ **시연영상 URL** (유튜브)
- [ ] **프로젝트 등록 URL**:
      `https://github.com/chjnett/ossp-2026-llm-router-challenge/tree/22178dd`
      → `git rev-parse 22178dd`로 전체 40자리 확인 후 `tree/<전체SHA>` 형식으로
- [ ] SBOM: `container/sbom.cdx.json` (108개) — 붙임1에 기재
- [ ] 붙임2: AI 모델 **"해당 없음"** + Claude·Codex 사용 범위 (REPORT_FORM.md에 이미 작성)
- [ ] 회색 가이드 문구 삭제, 5페이지 이내 확인
- [ ] 파일명: `2026 오픈소스 개발자대회 결과보고서_379(팀명)`

## 3. 제출 (osscontest.kr, 마감 전)

- [ ] **zip 압축**: ① 결과보고서 원본(HWPX/DOCX) ② PDF 변환 ③ 중복수혜 확인서(해당 시)
- [ ] 접수 및 조회 > 출품작 제출 > 제출하기 > 파일 업로드 > [출품작 제출 완료하기]
- [ ] **제출 완료 확인**: 화면 상태 "제출 완료" + 완료 안내 메일 수신 — 둘 다 확인 필수
- [ ] 마감 전 재업로드 가능 — **마감 시점 최종 제출본이 심사됨** (전략: `plan/submission/SUBMIT_EARLY.md`)

## 4. 이후 라우터를 개선해 재제출할 때

절차: `plan/submission/SUBMIT_EARLY.md` — 7단계 사이클
핵심:
1. 코드 커밋 → push (`git push origin main`)
2. 이미지 재빌드·push → 새 digest
3. `submission-ossp-skt.json` 갱신 (JSON만 별도 커밋)
4. `tools/verify_release.py --evidence build/measure/evidence.json` 9/9 통과 확인
5. 보고서 URL을 새 JSON 커밋의 `tree/<SHA>`로 바꿔 재업로드

## 참고 경로 (한눈에)

| 문서 | 경로 |
| --- | --- |
| 제출 TODO (전체) | `plan/submission/SUBMISSION_TODO.md` |
| 보고서 원고 (채울 것) | `plan/submission/REPORT_FORM.md` |
| 조기제출+지속수정 전략 | `plan/submission/SUBMIT_EARLY.md` |
| 데모 영상 스크립트 | `plan/submission/DEMO_SCRIPT.md` |
| 실측 수치 (보고서 근거) | `plan/REPORT.md` |
| 제출 절차 (명령) | `plan/RELEASE.md` |
| SBOM | `container/sbom.cdx.json` |
| 기술 제출 JSON (생성됨) | `submission-ossp-skt.json` (저장소 루트) |
| 검증 도구 | `tools/validate_technical_submission.py` · `tools/verify_release.py` · `tools/check_runtime.py` |
