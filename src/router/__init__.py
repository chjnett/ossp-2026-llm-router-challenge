# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""프롬프트 기반 LLM 라우터.

모듈 구성은 ``plan/DESIGN.md``의 6단계 계약을 따른다.

- ``data``      공개 Train/Dev를 실험용 배열로 올린다 (오프라인 전용)
- ``harness``   공식 채점기로 평가한다. 지름길 점수를 최종 지표로 쓰지 않는다
- ``allocate``  [A] λ 할당기와 [S] 안전 검증. 순서 불변성이 정확성 요건이다

런타임 진입점은 ``router-run``이며, 모델 선택 함수는 프롬프트 내용과 등급만
받는다. 문항 ID·입력 순서·프롬프트 밖 메타데이터는 선택에 쓰지 않는다.
"""
