<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 결과보고서 제출 참고 (2026 오픈소스 개발자대회)

공식 양식(osscontest.kr)에 옮겨 넣을 내용. **⬜ = 제출 전 채울 칸.**
분량 5페이지 이내, 맑은고딕 10pt, 회색 가이드 문구 삭제.
파일명: `2026 오픈소스 개발자대회 결과보고서_379(팀명)`
접수번호: **379**
마감: **2026-08-27(목) 18:00 KST**

> 관련 문서: 본문 근거는 `plan/REPORT.md`(실측 수치), `plan/DESIGN.md`(설계·판정),
> `plan/TODO.md`(현황), `plan/RULES.md`(규칙). SBOM은 `container/sbom.cdx.json`.

---

## 파레토 최적화 — 이 프로젝트의 핵심 차별점 (요약문)

> 본 라우터는 **"점수 최대화"와 "예산 파산 위험" 사이의 파레토 프론티어를 실측으로
> 매핑**하고, 기대값(EV)이 최고인 지점을 선택한다. 예산을 넘기면 그 등급은 0점이므로
> 파산 확률은 점수와 같은 차원의 목표다. 아래 표가 그 프론티어다.

| 프론티어 (최종 재적합, Dev 880) | 파산(스트레스) | 판정 |
|---|---|---|
| 보수: risk_quantile 0.80 | 0.681335 | 0회 |
| 절충: q 0.72 | 0.683210 | 2회 |
| **채택: q 0.68 + unseen_risk_boost** | **0.683892** | **26회 (0.217%)** |
| 과공격: q 0.65 | 0.684574 | 70회 (0.583%) |

기대값(EV) = 점수 − 파산확률 × 파산손실(가중 0.216). **q=0.68이 EV 최고점**이며,
`unseen_risk_boost`(미지 계열에만 +20% 보수 비용)로 "본 계열은 공격, 미지 계열은 보수"
를 분리해 leave-one-family-out 일반화(결합 Premium 3.60)를 지킨다. out-of-sample
(Train-only 적합 → Dev)로 환산하면 0.681420이며, 공개된 최강 참고 구현 대비
+0.008 수준이다.

---

# 2026년 오픈소스 개발자대회 결과보고서

## 표지

| 항목 | 내용 |
| --- | --- |
| 팀명 | ⬜ [접수 정보와 동일] |
| 팀 인원 (팀장 포함) | 1명 |
| 참가부문 | ⬜ [학생/일반] |
| 과제유형 | 지정과제 (SK텔레콤) |

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Efficient LLM Router — 예산 제약 하의 compute-optimal 프롬프트 라우팅** |
| 프로젝트 등록 URL | `https://github.com/chjnett/ossp-2026-llm-router-challenge` |
| 시연영상 | ⬜ [유튜브 URL] |
| 프로젝트 소개 | 프롬프트 하나마다 비용이 다른 세 LLM 중 하나를 골라, 등급별 예산 한도 안에서 정답률 합계를 최대화하는 오픈소스 라우터. 모델을 호출하지 않고 프롬프트 본문과 예산 등급만으로 결정한다. |

## 프로젝트 세부 내용

### 개발배경 및 목적

- 대규모 LLM 서비스에서 비싼 모델을 모든 요청에 쓰면 비용이 폭증한다. **요청마다
  난이도와 가치가 다르므로, 예산 안에서 가장 가성비 좋게 모델을 배분**하는 것이 실용적 과제다.
- 이 대회는 그 배분을 **"모델을 실행하지 않고 프롬프트 본문만 보고"** 해야 한다는
  극단적 제약을 건다. 즉 라우팅 품질을 예측 모델로 사전에 학습해야 한다.
- 목표: 예산 초과(해당 등급 0점) 없이 가중 정답률을 최대화하는 것.

### 개발환경

- 하드웨어: Apple Silicon(arm64) / CI: GitHub Actions `ubuntu-24.04-arm`
- SW: Python 3.11·3.14, NumPy 2.2.6(BSD-3), scikit-learn(학습 전용), Docker Buildx
- 배포: `linux/arm64` 컨테이너 (네트워크 없음, CPU 2, 메모리 2GiB, 90초)

### 시스템 구성 및 아키텍처

```
프롬프트 ─▶ [F] 특징 추출(해시 n-gram + 9개 문제 계열 분류)
            ├─▶ [Q] 점수 헤드   ŝ(light/ax31/K1)  — 강축소 ridge (alpha=32000)
            ├─▶ [C] 비용 헤드   ĉ(모델별 비용)     — tier별 risk_quantile
            └─▶ [G] 게이트       K1 꼬리·미지 계열 차단
                    │
        [A] ROI 배분: 이득/비용 내림차순으로 예산이 닿는 데까지 승격
                    │
        [S] 안전 검증: 더 비관적 비용으로 재검산, 초과 시 강등
                    ▼
        submission.json (모델 선택 목록)
```

- 핵심 데이터 흐름: 학습(공개 Train/Dev) → 아티팩트(능형회귀 계수 JSON) → 컨테이너가
  아티팩트를 읽고 추론만 수행(런타임 학습 없음).

### 프로젝트 주요기능 (프로젝트 상세 내용)

1. **예측**: 프롬프트에서 해시 n-gram 특징과 문제 계열을 뽑아, 세 모델의 품질·비용을
   ridge 회귀로 예측. 비용은 로그공간 예측의 Jensen 편향과 계열별 상방 분위수를 보정.
2. **배분**: 예측 이득/추가 비용(ROI) 순으로 예산이 닿는 데까지 승격. 입력 순서·문항
   ID와 무관하게 결정(감사 재실행 안전).
3. **안전**: 파산 게이트 — 분포 이동 6종 시나리오 × 배치 크기 3종 × 2,000회 스트레스,
   그리고 leave-one-family-out(계열 통째로 제외)로 미지 분포 일반화를 검증.
4. **파레토 EV 최적화 (차별점)**: 스트레스 파산률을 점수와 같은 축으로 두고 프론티어를
   실측. `unseen_risk_boost`로 "본 계열은 공격, 미지 계열은 보수"를 분리해, 공격적
   비용 설정에서도 미지 계열 일반화를 지켰다.

### 구동 및 시연

```console
docker build --platform linux/arm64 -f container/router.Dockerfile -t ossp-router .
for tier in fast balanced premium; do
  docker run --network none --cpus 2 --memory 2g -v $PWD/input:/challenge/input:ro \
    -v $PWD/out:/challenge/output ossp-router --tier $tier
done
PYTHONPATH=src python3 -m ossp_router.cli self-check \
  --input data/materialized/dev/inputs.json --outcomes data/dev/outcomes.json \
  --submissions build/out --report build/report.json
```

- 세 등급 실행: fast 2.6s / balanced 2.5s / premium 4.5s (한도 90초의 5%)
- 공식 채점 결과: **fast 0.655682 / balanced 0.684659 / premium 0.720739 / 가중 0.683892**
- 예산 사용률: 88.7% / 76.8% / 73.4% (모두 통과)

### 기대효과 및 활용분야

- LLM 서비스 비용 최적화의 실용적 참조 구현. 모델 호출 없이 프롬프트만으로 라우팅하는
  기법은 비용 민감 추론(cost-sensitive inference) 전반에 적용 가능.
- 향후: 더 정확한 출력 토큰(비용) 예측 모델로 교체하면 안전 상한(~0.70)까지 회수 가능.
  프롬프트 라우팅은 MaaS(model-as-a-service) 게이트웨이, RAG 파이프라인 등에 확장 가능.

### 기타 — 혁신성 및 차별성

- **파레토 프론티어 실측**: 점수-파산 위험 곡선을 직접 측정해 EV 최고점을 선택. 단순
  "안전 마진 확보"가 아니라 위험을 정량화하고 최적점을 수리적으로 고른 점이 차별점.
- **unseen_risk_boost**: "본 계열 공격 + 미지 계열 보수" 비용 분리. 이로써 공격적
  설정에서도 leave-one-family-out 일반화(결합 Premium 3.60 < 4.0)를 유지.
- **강한 축소 ridge**: alpha=32,000으로 lexical 과적합을 억제 → out-of-sample
  일반화가 공개 참고 구현(alpha=5, 18만 차원 TF-IDF)보다 +0.008 우월.
- **엄격한 검증**: 36,000+ 스트레스, LOFO 9/9, ID·순서 불변성, arm64 컨테이너 검증.

### 한계점 및 향후 발전 로드맵

- 한계: ① K1 비용 꼬리(출력 토큰 5~566배)로 비용 예측이 보수적 → 예산의 27% 미사용.
  ② 점수는 2~4회 생성 평균이라 예측 한계(R² 0.23~0.31). ③ 공개 Dev에 없는 미지
  계열은 어떤 게이트도 완전히 막지 못함.
- 로드맵: ① 출력 토큰 예측 모델 개선(트리·임베딩은 CV에서 기각했으나 다른 표현 탐색)
  ② 라우팅을 API 게이트웨이 형태로 오픈소스 배포 ③ 다국어·코드·수학 도메인 확장.

### 소감 및 후기

- 기술적 한계 극복: "표현을 바꾸면 뚫린다"는 가설을 임베딩·kNN·GBDT 등 7개 트랙으로
  검증했지만 모두 실측으로 기각. 점수 상승의 대부분은 구현 결함 수정(중복 프롬프트
  분리, Jensen 편향, 총합 보정)에서 나왔다 — "모델링보다 정확성이 이긴다"는 교훈.
- 파레토 프론티어라는 판정 프레임을 도입해, 안전과 점수의 갈등을 감정이 아니라
  실측 수치로 해결했다.

---

# 붙임1 — SBOM (소프트웨어 자재명세서)

전체 108개 구성요소: 기반 OS(Debian 12) 105개 + Python 2개 + 참가자 1개.
상세는 `container/sbom.cdx.json`(CycloneDX 1.5) 참조.

| 번호 | 라이브러리명 | 버전 | 라이선스 | 공식 저장소 URL | 사용 목적 |
| --- | --- | --- | --- | --- | --- |
| 1 | numpy | 2.2.6 | BSD-3-Clause | https://github.com/numpy/numpy | 라우터 예측·배분 연산 (유일한 결합 의존성) |
| 2 | Python (CPython) | 3.11.15 | PSF-2.0 | https://github.com/python/cpython | 실행 런타임 |
| 3 | base image (Debian 12 slim) | bookworm | Debian-2 | https://www.debian.org/ | 컨테이너 기반 OS |
| 4~108 | 기반 이미지 내 OS 패키지 105종 | — | 각 패키지 copyright | https://www.debian.org/ | OS 표준 구성요소 |

※ 라이선스 근거(105개 OS 패키지)는 이미지 내 `/usr/share/doc/<패키지>/copyright`에 보존.
※ 참가자 작성 코드는 Apache-2.0.

---

# 붙임2 — AI 모델 활용 및 라이선스 기술 명세서

## 1. AI 모델 활용 유형

**□ 해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음**

> 라우터는 능형회귀 계수(85.9 KB JSON)만 사용한다. 신경망 가중치, 사전학습 모델,
> 토크나이저를 포함하지 않는다. 실행 중 어떤 모델도 호출하지 않는다.
> (규칙: `CHALLENGE_RULES.md` "실행 이미지에 AI 모델을 포함하지 않은 라우터는
> '해당 없음'으로 밝힌다.")

## 2. 기반(베이스) 모델 정보

**해당 없음** — 기반 모델을 사용하지 않음 (유형 3도 아님: 학습 산출물은 선형 회귀 계수).

## 3. 데이터셋 정보 및 가중치 배포 명세

**해당 없음** — 신경망 가중치를 생성하지 않음. 능형회귀 계수는 공개 Train/Dev(1,760+880)
로 학습하며, 계수 자체는 `src/router/resources/artifact.v1.json`에 Apache-2.0으로 포함.

## 4. 소스코드 라이선스 및 개발 환경 정보

| 항목 | 내용 |
| --- | --- |
| 직접 작성한 코드의 오픈소스 라이선스 | **Apache-2.0** |
| 학습/추론 소스코드 공개 저장소 URL | `https://github.com/chjnett/ossp-2026-llm-router-challenge` |
| 상용 AI 보조도구 활용 여부 및 범위 | Anthropic Claude Code(Claude Opus)·OpenAI Codex를 코드 작성·실험 설계 보조로 활용. 최종 판단·실험 채택·제출 결정은 참가자가 수행. 생성 코드는 참가자가 검토, 381개 테스트와 공식 채점기로 검증 |

---

## 제출 전 체크리스트

- [ ] 팀명·접수번호·참가부문 기재
- [ ] 프로젝트 등록 URL = `submission-ossp-skt.json`을 담은 **커밋 스냅샷** URL
- [ ] 시연영상 유튜브 URL 기재
- [ ] SBOM = `container/sbom.cdx.json` (108개)
- [ ] 붙임2: 해당 없음 + 상용 AI 도구 범위 기재
- [ ] 5페이지 이내, 맑은고딕 10pt, 회색 문구 삭제
- [ ] HWP/DOC 1부 + PDF 1부, 마감(8/27 18:00) 전 제출
