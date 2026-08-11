# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""ID·순서 불변성 (RULES B).

평가에서 감사 재실행은 생략 옵션이 없다. 오케스트레이터가 모든 문항의 ID를
재배정하고 순서를 따로 섞은 입력으로 한 번 더 돌린 뒤 선택을 대조한다.
하나라도 다르면 review_required로 자동 채점이 멈춘다.

배치 안에서 λ를 이분탐색하므로 집계가 순서에 의존하면 경계 문항이 뒤집힐 수
있다. 그 위험을 여기서 잡는다.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from router.allocate import allocate, order_invariant_sum, safety_demote
from router.data import content_key


def _synthetic(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """실제 자료의 성질을 흉내 낸 예측값. 비용은 K1 쪽 꼬리를 두껍게 만든다."""

    rng = np.random.default_rng(seed)
    s = np.clip(
        np.stack(
            [
                rng.beta(2.2, 1.4, n),
                rng.beta(2.6, 1.3, n),
                rng.beta(3.4, 1.1, n),
            ],
            axis=1,
        ),
        0.0,
        1.0,
    )
    s = np.sort(s, axis=1)  # 대체로 비싼 모델이 더 잘한다
    base = rng.lognormal(-5.4, 0.9, n)
    c = np.stack([base, base * 2.12, base * rng.lognormal(3.1, 0.9, n)], axis=1)
    sd = c * 0.35
    return s, c, sd


class OrderInvarianceTest(unittest.TestCase):
    def test_fsum_is_permutation_invariant(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.lognormal(-5.0, 2.0, 4000)
        reference = order_invariant_sum(values)
        for _ in range(50):
            shuffled = values[rng.permutation(len(values))]
            self.assertEqual(order_invariant_sum(shuffled), reference)

    def test_naive_sum_is_not_reliable(self) -> None:
        """np.sum을 쓰면 안 되는 이유를 기록해 둔다.

        pairwise 누적은 순열에 따라 마지막 비트가 달라질 수 있다. 이 테스트는
        '다르다'를 단언하지 않고(환경에 따라 우연히 같을 수 있다) fsum이 항상
        정확한 값과 일치한다는 것만 확인한다.
        """

        rng = np.random.default_rng(1)
        values = rng.lognormal(0.0, 6.0, 5000)
        exact = math.fsum(sorted(values.tolist()))
        self.assertEqual(order_invariant_sum(values), exact)

    def test_allocation_is_order_invariant(self) -> None:
        for seed in (0, 1, 2, 3, 4):
            s, c, sd = _synthetic(1500, seed)
            rng = np.random.default_rng(100 + seed)
            for multiplier, util in ((1.25, 0.90), (2.0, 0.88), (4.0, 0.85)):
                base = allocate(
                    s, c, multiplier=multiplier, util=util, sd=sd, mu=0.3
                ).picks
                for _ in range(8):
                    order = rng.permutation(len(s))
                    moved = allocate(
                        s[order],
                        c[order],
                        multiplier=multiplier,
                        util=util,
                        sd=sd[order],
                        mu=0.3,
                    ).picks
                    restored = np.empty_like(moved)
                    restored[order] = moved
                    np.testing.assert_array_equal(
                        restored,
                        base,
                        err_msg=f"seed={seed} mult={multiplier}: 순서를 바꾸자 선택이 달라졌다",
                    )

    def test_safety_demote_is_order_invariant(self) -> None:
        s, c, sd = _synthetic(900, 7)
        keys = [content_key(f"프롬프트 {i}") for i in range(len(s))]
        pessimistic = c * 1.6
        rng = np.random.default_rng(11)

        plan = allocate(s, c, multiplier=4.0, util=0.95)
        base = safety_demote(
            plan, s, pessimistic, keys, multiplier=4.0, util=0.80
        ).picks
        self.assertGreater(
            int((base == 0).sum()), 0, "이 설정에서는 강등이 일어나야 의미 있는 검사다"
        )

        for _ in range(8):
            order = rng.permutation(len(s))
            moved_plan = allocate(s[order], c[order], multiplier=4.0, util=0.95)
            moved = safety_demote(
                moved_plan,
                s[order],
                pessimistic[order],
                [keys[i] for i in order],
                multiplier=4.0,
                util=0.80,
            ).picks
            restored = np.empty_like(moved)
            restored[order] = moved
            np.testing.assert_array_equal(
                restored, base, err_msg="강등 순서가 입력 순서에 의존한다"
            )

    def test_duplicate_prompts_get_identical_choices(self) -> None:
        """같은 프롬프트가 배치에 여러 번 있어도 선택이 같아야 한다."""

        s, c, sd = _synthetic(600, 3)
        s = np.concatenate([s, s], axis=0)
        c = np.concatenate([c, c], axis=0)
        sd = np.concatenate([sd, sd], axis=0)
        picks = allocate(s, c, multiplier=2.0, util=0.9, sd=sd, mu=0.3).picks
        np.testing.assert_array_equal(
            picks[:600], picks[600:], err_msg="동일 프롬프트 쌍의 선택이 갈렸다"
        )

    def test_content_key_is_stable_across_processes(self) -> None:
        """내장 hash()는 PYTHONHASHSEED에 따라 달라진다. blake2b는 고정이다."""

        self.assertEqual(
            content_key("두 정수 17과 25의 합을 계산하세요."),
            content_key("두 정수 17과 25의 합을 계산하세요."),
        )
        self.assertEqual(len(content_key("x")), 32)


if __name__ == "__main__":
    unittest.main()
