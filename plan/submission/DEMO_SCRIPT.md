<!--
SPDX-FileCopyrightText: Copyright 2026 chjnett
SPDX-License-Identifier: Apache-2.0
-->

# 데모 영상 스크립트 (2분 45초)

요구사항: 2~3분, 유튜브 URL(`docs/SUBMISSION.md` "데모 영상 URL 기재").
촬영: 화면 녹화(OBS 등) + 나레이션. 터미널은 어두운 테마, 글자 크게.

## 씬 구성

| # | 시간 | 화면 | 나레이션 (한글) |
| --- | --- | --- | --- |
| 1 | 0:00-0:15 | 타이틀 카드: "Efficient LLM Router — 예산 제약 하의 compute-optimal 프롬프트 라우팅" | "프롬프트마다 예산에 가장 잘 맞는 LLM을 고르는 오픈소스 라우터입니다. 모델을 실행하지 않고, 프롬프트 본문과 예산 등급만 보고 결정합니다." |
| 2 | 0:15-0:35 | 세 모델 비용 그래프 (light 1×, ax31 2.1×, K1 23.8×) + 등급 표 (Fast 1.25×/40%, Balanced 2×/30%, Premium 4×/30%) | "세 모델은 최대 23배의 비용 차이가 납니다. 그리고 등급마다 예산이 정해져 있어, 예산을 한 번이라도 넘기면 그 등급은 0점이 됩니다. 그래서 어느 문제에 비싼 모델을 쓸지가 전부입니다." |
| 3 | 0:35-1:00 | 파레토 프론티어 그래프 (점수 vs 파산위험, q 0.8→0.65 곡선) + `unseen_risk_boost` 도식 | "우리는 점수와 파산 위험 사이의 파레토 프론티어를 실측으로 그리고, 기대값이 최고인 지점을 선택했습니다. 핵심은 unseen_risk_boost — 본 계열은 공격적으로, 미지 계열은 보수적으로 비용을 평가해, 공격적 설정에서도 일반화를 지킵니다." |
| 4 | 1:00-1:30 | 터미널: `docker run --network none --cpus 2 --memory 2g` 3등급 실행 → `submission.json` 생성 | "공식 컨테이너로 세 등급을 실행합니다. 네트워크 없이, CPU 2개와 2GB 메모리 안에서. 880문항을 5초 안에 처리해 선택 결과 JSON을 만듭니다." |
| 5 | 1:30-2:00 | 터미널: `self-check` → 등급별 점수·예산 사용률 표 | "공식 채점기로 검증합니다. fast 0.656, balanced 0.686, premium 0.721, 가중 최종 0.684. 세 등급 모두 예산을 통과했습니다." |
| 6 | 2:00-2:25 | 터미널: `tools/verify_release.py` → 9개 항목 전부 ✓ | "마지막으로 제출 관문을 확인합니다. 커밋과 이미지가 맞물리고, 이미지 크기와 VOLUME, 산출물 일치 — 9개 항목이 전부 통과했습니다." |
| 7 | 2:25-2:45 | GitHub 저장소 페이지 + Apache-2.0 배지 | "Apache-2.0 오픈소스로 공개되어 있습니다. 저장소에서 전체 코드, 학습 절차, 검증 도구를 확인할 수 있습니다. 감사합니다." |

## 촬영 준비 (명령)

```bash
# 1) 이미지 (이미 빌드돼 있음: ghcr.io/chjnett/ossp-router:submission)
docker pull ghcr.io/chjnett/ossp-router@sha256:970e504f67f02371ce71393818df2855563a701f1793d1e0984902c5d4e5f4fb

# 2) 씬 4~6 일괄 실행. 등급별 산출물과 촬영용 로그를 build/demo/에 저장
sh plan/submission/run_demo.sh
```

`run_demo.sh`는 변경 가능한 태그가 아니라 제출 다이제스트를 실행하며, 등급별
`submission.json`을 `build/demo/submissions/{fast,balanced,premium}.json`으로
분리한 뒤 공식 self-check와 이미지 크기 증거, 9개 제출 관문까지 연속으로
검증합니다.

## 자동 렌더링 (초안 MP4)

씬 4~6 실측이 끝난 뒤 macOS에서 다음 명령을 실행하면 기존 그림, 실제 로그,
한국어 Yuna TTS를 합쳐 1920×1080 영상 초안을 만듭니다.

```bash
python3 tools/render_demo_video.py
```

출력: `build/demo/Efficient-LLM-Router-OSSP-2026.mp4` (2분 45초)

TTS 음성은 업로드 가능한 초안이며, 본인 목소리를 선호하면 같은 타임라인으로
나레이션만 교체하면 됩니다.

## 대본 참고 (표시용 수치)

- 가중 점수 0.6844 (fast 0.6557 · balanced 0.6864 · premium 0.7207) — **t43 실측 반올림**
- 예산 사용률: 88.7% / 77.7% / 73.4% — 모두 한도 내
- 실행: 880문항 fast 2.6s · balanced 2.5s · premium 4.5s (한도 90s의 5%)
- 이미지 크기: 압축 63.4 MiB (한도 1024) · rootfs 196 MiB (한도 2048)
- 파레토: 보수 q0.8(0.681) → 채택 t43(0.684) → 과공격 q0.65(0.685) 곡선

## 업로드

- 유튜브 업로드 (비공개 → 공개, 제목: "Efficient LLM Router — OSSP 2026")
- `REPORT_FORM.md`의 "시연영상" 칸에 URL 기재
- 데모 영상 URL은 결과보고서 필수 기재 항목 (`docs/SUBMISSION.md`)
