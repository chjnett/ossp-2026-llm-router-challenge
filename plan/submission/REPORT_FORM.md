<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 2026년 오픈소스 개발자대회 결과보고서 (완성본)

> 이 문서는 공식 양식의 각 칸에 들어갈 내용을 **전부 채운** 버전이다.
> HWP/DOCX 양식에 복사해 넣고, ⬜ 칸(팀 정보·시연영상 URL)만 채우면 된다.
> 마감: 2026-08-27(목) 18:00 · 접수번호: 379
> 제출: 원본 1부 + PDF 1부를 **zip 압축**해 osscontest.kr 업로드.
>
> **시각 자료** (아래 "그림 삽입" 표시된 곳에 삽입, 파일은
> `plan/submission/figures/`에 있음):
> - `figures/pareto-frontier.png` — 점수-파산위험 파레토 프론티어 (핵심 차별점)
> - `figures/budget-usage.png` — 등급별 예산 사용률
> - `figures/model-cost.png` — 모델별 비용 구조
> - `figures/tier-scores.png` — 등급별 점수
> - `figures/model-choices.png` — 등급별 모델 선택 분포
> - `figures/family-gain.png` — 문제 계열별 K1 이득
> - `figures/k1-cost-tail.png` — K1 비용 꼬리 분포
> - `figures/champion-progress.png` — 챔피언 점수 진행 (EV 최적점 선택)

---

## 표지

| 항목 | 내용 |
| --- | --- |
| 팀명 | ⬜ [접수 정보와 동일] |
| 팀 인원 (팀장 포함) | 1명 |
| 참가부문 | ⬜ [학생/일반] |
| 과제유형 | 지정과제 (SK텔레콤) |

---

## □ 결과보고서

### 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | **Efficient LLM Router — 예산 제약 하의 compute-optimal 프롬프트 라우팅** |
| 프로젝트 등록 URL | `https://github.com/chjnett/ossp-2026-llm-router-challenge` |
| 시연영상 | ⬜ [유튜브 URL — 3분 이내] |
| 프로젝트 소개 | 프롬프트 하나마다 비용이 다른 세 LLM 중 하나를 골라, 등급별 예산 한도 안에서 정답률 합계를 최대화하는 오픈소스 라우터. 모델을 호출하지 않고 프롬프트 본문과 예산 등급만으로 결정한다. |

### 프로젝트 세부 내용

#### 개발배경 및 목적

- 대규모 LLM 서비스에서 비싼 모델을 모든 요청에 쓰면 비용이 폭증한다. 요청마다
  난이도와 가치가 다르므로, **예산 안에서 가장 가성비 좋게 모델을 배분**하는 것이
  실용적 과제다.
- 세 모델의 비용은 최대 23.8배 차이(K1이 평균 23.8배, 문항별 5~566배)로, 비싼
  모델을 언제 쓸지가 곧 비용 전체를 결정한다.

**[그림 삽입: figures/model-cost.png — 모델별 비용 구조]**
**[그림 삽입: figures/k1-cost-tail.png — K1 비용 꼬리]** (한계·안전설계 근거)
- 이 대회는 그 배분을 **"모델을 실행하지 않고 프롬프트 본문만 보고"** 해야 한다는
  극단적 제약을 건다. 즉 라우팅 품질을 예측 모델로 사전에 학습해야 한다.
- 목표: 예산 초과(해당 등급 0점) 없이 가중 정답률을 최대화하는 것.

#### 개발환경

- 하드웨어: Apple Silicon(arm64), CI: GitHub Actions `ubuntu-24.04-arm`
- SW: Python 3.11·3.14, NumPy 2.2.6(BSD-3), scikit-learn(학습 전용), Docker Buildx
- 배포: `linux/arm64` 컨테이너 — 네트워크 없음, CPU 2, 메모리 2GiB, 등급당 90초
- 라이선스: Apache-2.0, REUSE 3.3 준수

#### 시스템 구성 및 아키텍처

```
프롬프트 ─▶ [F] 특징 추출(해시 n-gram + 9개 문제 계열 분류)
            ├─▶ [Q] 점수 헤드   ŝ(light/ax31/K1) — 강축소 ridge (alpha=32,000)
            ├─▶ [C] 비용 헤드   ĉ(모델별 비용) — tier별 risk_quantile + unseen_risk_boost
            └─▶ [G] 게이트       K1 꼬리·미지 계열 차단
                    │
        [A] ROI 배분: 이득/비용 내림차순으로 예산이 닿는 데까지 승격
                    │
        [S] 안전 검증: 더 비관적 비용으로 재검산, 초과 시 강등
                    ▼
        submission.json (모델 선택 목록)
```

- 데이터 흐름: 공개 Train/Dev 학습 → 아티팩트(능형회귀 계수 JSON) → 컨테이너가
  아티팩트를 읽고 추론만 수행(런타임 학습 없음).

#### 프로젝트 주요기능 (상세)

1. **예측**: 프롬프트에서 해시 n-gram 특징과 문제 계열을 뽑아 세 모델의 품질·비용을
   ridge 회귀로 예측. 비용은 로그-지수 변환의 Jensen 편향과 계열별 상방 분위수를 보정.
2. **배분**: 예측 이득/추가 비용(ROI) 순으로 예산이 닿는 데까지 승격. 입력 순서·문항
   ID와 무관하게 결정(감사 재실행 안전).
3. **안전**: 파산 게이트 — 분포 이동 6종 시나리오 × 배치 크기 3종 × 2,000회 스트레스,
   그리고 leave-one-family-out(계열 통째로 제외)로 미지 분포 일반화를 검증.
4. **파레토 EV 최적화 (차별점)**: 스트레스 파산률을 점수와 같은 축으로 두고
   프론티어를 실측. `unseen_risk_boost`로 "본 계열은 공격, 미지 계열은 보수"를 분리해,
   공격적 비용 설정에서도 미지 계열 일반화(결합 Premium 3.60 < 4.0)를 유지.

**[그림 삽입: figures/pareto-frontier.png — 점수-파산위험 파레토 프론티어]**
프론티어에서 기대값(EV)이 최고인 지점(q=0.68)을 채택했다. q=0.65로 더 공격적으로
가도 EV는 같아지고 위험만 커진다.
**[그림 삽입: figures/family-gain.png — 문제 계열별 K1 이득]**
**[그림 삽입: figures/champion-progress.png — 챔피언 점수 진행]**

#### 구동 및 시연

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

| 항목 | 결과 |
| --- | --- |
| 실행 (2,640문항, 3등급) | fast 7.7s · balanced 7.2s · premium 18.2s (한도 90s) |
| 공식 채점 | fast 0.655682 · balanced 0.684659 · premium 0.720739 · **가중 0.683892** |
| 예산 사용률 | 88.7% / 76.8% / 73.4% — 모두 한도 내 |
| 제출 관문 | `verify_release.py` 9항목 전부 통과 |

**[그림 삽입: figures/budget-usage.png — 등급별 예산 사용률]**
**[그림 삽입: figures/tier-scores.png — 등급별 점수]**
**[그림 삽입: figures/model-choices.png — 등급별 모델 선택 분포]**

#### 기대효과 및 활용분야

- LLM 서비스 비용 최적화의 실용적 참조 구현. 모델 호출 없이 프롬프트만으로 라우팅하는
  기법은 비용 민감 추론(cost-sensitive inference) 전반에 적용 가능.
- 향후: 더 정확한 출력 토큰(비용) 예측 모델로 교체하면 안전 상한(~0.70)까지 회수 가능.
  프롬프트 라우팅은 MaaS 게이트웨이, RAG 파이프라인, 멀티모델 서비스에 확장 가능.

#### 기타 — 혁신성 및 차별성

- **파레토 프론티어 실측**: 점수-파산 위험 곡선을 직접 측정해 기대값(EV) 최고점을
  선택. "안전 마진 확보"가 아니라 위험을 정량화하고 최적점을 수리적으로 고른 점이 차별점.
- **unseen_risk_boost**: "본 계열 공격 + 미지 계열 보수" 비용 분리. 공격적 설정에서도
  leave-one-family-out 일반화 유지.
- **강한 축소 ridge**: alpha=32,000으로 lexical 과적합 억제 → out-of-sample 일반화가
  공개 참고 구현보다 우월.
- **엄격한 검증**: 72,000 스트레스, LOFO 9/9, ID·순서 불변성, arm64 컨테이너 검증.

#### 한계점 및 향후 발전 로드맵

- 한계: ① K1 비용 꼬리(출력 토큰 5~566배)로 비용 예측이 보수적 → 예산의 27% 미사용.
  ② 점수는 2~4회 생성 평균이라 예측 한계(R² 0.23~0.31). ③ 미지 계열은 어떤 게이트도
  완전히 막지 못함.
- 로드맵: ① 출력 토큰 예측 모델 개선(트리·임베딩은 CV에서 기각, 다른 표현 탐색)
  ② 라우팅을 API 게이트웨이 형태로 오픈소스 배포 ③ 다국어·코드·수학 도메인 확장.

#### 소감 및 후기

- "표현을 바꾸면 뚫린다"는 가설을 임베딩·kNN·GBDT 등 7개 트랙으로 검증했지만 모두
  실측으로 기각. 점수 상승의 대부분은 구현 결함 수정(중복 프롬프트 분리, Jensen 편향,
  총합 보정)에서 나왔다 — "모델링보다 정확성이 이긴다"는 교훈.
- 파레토 프론티어라는 판정 프레임을 도입해, 안전과 점수의 갈등을 감정이 아니라 실측
  수치로 해결했다.

---

## 붙임1 — SBOM (소프트웨어 자재명세서)

전체 108개 구성요소: 기반 OS(Debian 12) 105개 + Python 2개 + 참가자 1개.
상세: `container/sbom.cdx.json` (CycloneDX 1.5).

| 번호 | 라이브러리명 | 버전 | 라이선스 | 공식 저장소 URL | 사용 목적 |
| --- | --- | --- | --- | --- | --- |
| 1 | numpy | 2.2.6 | BSD-3-Clause | https://github.com/numpy/numpy | 라우터 예측·배분 연산 (유일한 결합 의존성) |
| 2 | Python (CPython) | 3.11.15 | PSF-2.0 | https://github.com/python/cpython | 실행 런타임 |
| 3 | Debian 12 slim (base image) | bookworm | Debian-2 | https://www.debian.org/ | 컨테이너 기반 OS |
| 4~108 | 기반 이미지 내 OS 패키지 105종 | — | 각 copyright | https://www.debian.org/ | OS 표준 구성요소 |

※ 105개 OS 패키지의 라이선스 근거는 이미지 내 `/usr/share/doc/<패키지>/copyright`에 보존.

---

## 붙임2 — AI 모델 활용 및 라이선스 기술 명세서

### 1. AI 모델 활용 유형

**□ 해당 없음 — 실행 이미지에 AI 모델을 탑재하지 않음**

> 라우터는 능형회귀 계수(85.9 KB JSON)만 사용한다. 신경망 가중치, 사전학습 모델,
> 토크나이저를 포함하지 않는다. 실행 중 어떤 모델도 호출하지 않는다.

### 2. 기반(베이스) 모델 정보

**해당 없음** — 기반 모델을 사용하지 않음.

### 3. 데이터셋 정보 및 가중치 배포 명세

**해당 없음** — 신경망 가중치를 생성하지 않음. 능형회귀 계수는 공개 Train/Dev
(1,760 + 880)로 학습하며, 계수 자체는 `src/router/resources/artifact.v1.json`에
Apache-2.0으로 포함.

### 4. 소스코드 라이선스 및 개발 환경 정보

| 항목 | 내용 |
| --- | --- |
| 직접 작성한 코드의 오픈소스 라이선스 | **Apache-2.0** |
| 학습/추론 소스코드 공개 저장소 URL | `https://github.com/chjnett/ossp-2026-llm-router-challenge` |
| 상용 AI 보조도구 활용 여부 및 범위 | Anthropic Claude Code(Claude Opus)·OpenAI Codex를 코드 작성·실험 설계 보조로 활용. 최종 판단·실험 채택·제출 결정은 참가자가 수행. 생성 코드는 참가자가 검토, 396개 테스트와 공식 채점기로 검증 |

---

## 제출 파일 구성 (zip 압축)

```
2026 오픈소스 개발자대회 결과보고서_379(팀명).zip
├── ① 결과보고서 원본 (HWPX/DOCX)
├── ② 결과보고서 PDF 변환 파일
└── ③ 출품작 중복수혜 여부 확인서 (해당 시)
```

제출 전 체크: ① 시연영상 3분 이내·유튜브 URL 기재 ② SBOM 포함 ③ 붙임2는
"해당 없음"이라 **양식에서 삭제해도 됨** (가이드: "해당 사항이 없을 경우 삭제 후 제출")
④ 회색 문구 삭제·맑은고딕 10pt·5페이지 이내 ⑤ 마감(8/27 18:00) 전 "제출 완료"
상태 + 완료 메일 확인.
