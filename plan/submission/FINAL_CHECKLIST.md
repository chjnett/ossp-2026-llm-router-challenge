<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 최종 제출 체크리스트

마감: **2026-08-27(목) 18:00 KST** · 접수번호: **379**

이 문서는 `docs/SUBMISSION.md`, `docs/ENFORCEMENT.md`, `docs/RUNTIME.md`,
`plan/submission/README.md`, `SUBMISSION_TODO.md`, `REPORT_FORM.md`를 현재
저장소 상태와 대조한 실행용 체크리스트다.

## 1. 현재 동결 제출물 — 완료

- [x] 챔피언: `t43-bal-h112`
- [x] 공개 Dev 공식 채점: Fast `0.6557` · Balanced `0.6864` · Premium
      `0.7207` · 가중 `0.6844`
- [x] 예산 사용률: `88.7% / 77.7% / 73.4%`, 전 등급 통과
- [x] 이미지 빌드 코드 커밋:
      `35a8398d9316c9a7deef7d9d874b72aa86648309`
- [x] 공개 `linux/arm64` 이미지:
      `ghcr.io/chjnett/ossp-router@sha256:970e504f67f02371ce71393818df2855563a701f1793d1e0984902c5d4e5f4fb`
- [x] `submission-ossp-skt.json` 6필드 공식 스키마 통과
- [x] `verify_release.py` 9/9 통과: 공개 커밋·이미지·VOLUME·코드 일치·산출물·크기
- [x] 이미지 크기: 압축 `63.4 MiB` / rootfs `196.0 MiB`, 한도 내
- [x] 데모 영상 MP4 생성: `build/demo/Efficient-LLM-Router-OSSP-2026.mp4`
      (`1920×1080`, `2분 45초`, 한국어 나레이션)
- [x] SBOM 존재: `container/sbom.cdx.json` (CycloneDX 1.5, 108개 구성요소)

## 2. 결과보고서 수정 — 제출 전 필수

- [ ] 공식 HWP/HWPX 또는 DOCX 결과보고서 양식을 대회 사이트에서 다운로드
- [ ] 팀명 기재 — 접수 정보와 철자까지 동일하게
- [ ] 참가부문 기재 — 학생/일반 중 접수 정보와 동일하게
- [ ] 팀 인원 확인 — 현재 원고는 `1명`
- [ ] 프로젝트 등록 URL을 움직이는 브랜치나 저장소 기본 URL이 아닌 아래 고정
      스냅샷으로 기재

      `https://github.com/chjnett/ossp-2026-llm-router-challenge/tree/fbf73c0aa62a9d5da164c70d8840ac95e7d3b29d`

- [ ] 데모 영상을 유튜브에 업로드하고 `시연영상` 칸에 URL 기재
- [ ] 첫 페이지 작성 안내와 모든 회색 가이드 문구 삭제
- [ ] 본문 5페이지 이내·맑은고딕 10pt 확인
- [ ] 붙임1 SBOM 내용과 `container/sbom.cdx.json` 일치 확인
- [ ] 붙임2 AI 모델은 `해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음`
- [ ] AI 코딩 도구 사용 범위(Claude Code·OpenAI Codex) 유지

## 3. 그림 최신화 — 현재 최우선 차단 항목

다음 그림은 보고서 표와 달리 T38 또는 t43 이전 수치를 담고 있다. **현재 파일을
그대로 결과보고서에 넣지 않는다.** t43 실측으로 다시 만들거나 해당 그림을
보고서에서 제외한다.

- [ ] `figures/budget-usage.png`: Balanced `76.8%` → 실제 `77.7%`
- [ ] `figures/tier-scores.png`: Balanced `0.6847` → 실제 `0.6864`
- [ ] `figures/model-choices.png`: Balanced `ax31-light 119 / ax31 759` → 실제
      `107 / 771` (`axk1-think 2`)
- [ ] `figures/champion-progress.png`: T38에서 끝남 → t43 `0.6844` 추가
- [x] `figures/pareto-frontier.png`: `T38 champion` 표기를 제거하고 t43 채택·C4 통과 상태로 수정
- [ ] 수정한 모든 그림의 숫자를 `build/demo/report.json`과 한 번 더 대조

## 4. 영상 — 사용자 작업

- [ ] 최종 MP4를 처음부터 끝까지 직접 재생해 발음·화면 전환 확인
- [ ] 유튜브 제목: `Efficient LLM Router — OSSP 2026`
- [ ] 공개 또는 링크가 있는 사용자 공개로 업로드
- [ ] 업로드 후 재생시간이 3분 이내인지 확인
- [ ] 로그아웃 또는 시크릿 창에서 영상 URL이 열리는지 확인
- [ ] 확정 URL을 결과보고서 원본과 PDF 양쪽에 동일하게 반영

## 5. GitHub·이미지 공개 확인

- [ ] 로그아웃 상태에서 위 `tree/fbf73c0...` URL이 열리는지 확인
- [ ] 해당 스냅샷 루트에 `submission-ossp-skt.json`이 보이는지 확인
- [ ] `submission-ossp-skt.json`의 `commit_sha`가 `35a8398...`인지 확인
- [ ] 이미지 참조가 태그가 아니라 전체 `@sha256:970e...`인지 확인
- [ ] 로그아웃 상태에서 GHCR 이미지를 pull할 수 있는지 최종 확인
- [ ] 저장소를 평가 종료까지 public으로 유지
- [ ] 수상 시 저장소를 수상일로부터 5년 동안 public으로 유지

현재 로컬 `main`은 `origin/main`보다 데모 문서·렌더러 커밋 3개 앞서 있다.
이는 라우터 이미지나 동결 제출 스냅샷에는 영향을 주지 않는다. 이 커밋들을
push하더라도 `submission-ossp-skt.json`의 `commit_sha`와 이미지 digest를
현재 HEAD로 바꾸지 않는다. 바꾸려면 이미지를 새 커밋에서 다시 빌드해야 한다.

## 6. 제출 파일 만들기

- [ ] 결과보고서 원본 저장: HWPX/HWP 또는 DOCX
- [ ] 같은 내용으로 PDF 변환
- [ ] 원본과 PDF의 페이지·표·그림·URL이 동일한지 확인
- [ ] 중복수혜 확인서가 필요한 경우만 포함
- [ ] 다음 이름으로 ZIP 생성

      `2026 오픈소스 개발자대회 결과보고서_379(팀명).zip`

- [ ] ZIP을 다시 열어 아래 파일이 정상적으로 열리는지 확인

      1. 결과보고서 원본
      2. 결과보고서 PDF
      3. 중복수혜 확인서(해당 시)

## 7. osscontest.kr 최종 제출

- [ ] `접수 및 조회 → 출품작 제출 → 제출하기`에서 ZIP 업로드
- [ ] 업로드한 파일명이 접수번호 `379`와 팀명을 포함하는지 확인
- [ ] `출품작 제출 완료하기` 버튼까지 누르기
- [ ] 사이트 상태가 **제출 완료**인지 확인
- [ ] 제출 완료 안내 메일 수신 확인
- [ ] 완료 화면과 접수 시각을 스크린샷으로 보관
- [ ] 가능하면 **2026-08-27 17:30 KST 이전** 최종 업로드
- [ ] 재업로드했다면 마지막 업로드 파일이 최종본인지 다시 확인

## 8. 제출 직전 5분 점검

- [ ] 프로젝트 URL은 전체 40자리 커밋 스냅샷이다
- [ ] 유튜브 URL은 외부에서 열리고 3분 이내다
- [ ] 보고서 수치는 `0.6557 / 0.6864 / 0.7207 / 0.6844`다
- [ ] 예산 사용률은 `88.7% / 77.7% / 73.4%`다
- [ ] 오래된 T38 그림이나 `76.8%`, `0.6847`, `759` 수치가 남아 있지 않다
- [ ] 원본과 PDF의 본문이 5페이지 이내이며 회색 안내 문구가 없다
- [ ] ZIP 내부 파일이 모두 열린다
- [ ] 사이트 제출 완료와 완료 메일을 모두 확인했다

## 제출 완료 기준

아래 네 가지가 모두 충족돼야 실제 완료다.

- [ ] 공개 GitHub 고정 스냅샷
- [ ] 공개 GHCR 이미지 전체 digest
- [ ] 유튜브 URL을 포함한 결과보고서 원본 + PDF
- [ ] osscontest.kr `제출 완료` 상태 + 완료 메일
