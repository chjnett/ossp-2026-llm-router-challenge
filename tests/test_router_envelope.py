# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""오목 포락선 할당기의 성질 검사.

이 할당기는 예산 판정 경로 전체를 책임진다. 여기서 틀리면 등급이 0점이 되거나
감사 재실행에서 선택이 갈린다. 그래서 성질을 하나씩 못 박아 둔다.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from router.allocate import allocate, allocate_bisect
from router.data import content_key
from router.envelope import build_segments


def synthetic(n: int, seed: int):
    rng = np.random.default_rng(seed)
    s = np.sort(
        np.stack(
            [rng.beta(2.2, 1.4, n), rng.beta(2.6, 1.3, n), rng.beta(3.4, 1.1, n)],
            axis=1,
        ),
        axis=1,
    )
    base = rng.lognormal(-5.4, 0.9, n)
    c = np.stack([base, base * 2.12, base * rng.lognormal(3.1, 0.9, n)], axis=1)
    # 드물게 ax31이 light보다 싼 문항을 섞는다 (실제 자료에 1.1% 존재).
    cheap = rng.random(n) < 0.02
    c[cheap, 1] = c[cheap, 0] * 0.7
    keys = [content_key(f"p{i}-{seed}") for i in range(n)]
    return s, c, keys


class EnvelopeStructureTest(unittest.TestCase):
    def test_base_model_is_cheapest_allowed(self) -> None:
        s, c, keys = synthetic(400, 1)
        seg = build_segments(s, c, keys)
        np.testing.assert_array_equal(seg.base_model, np.argmin(c, axis=1))

    def test_segments_are_strictly_improving(self) -> None:
        s, c, keys = synthetic(400, 2)
        seg = build_segments(s, c, keys)
        self.assertTrue((seg.delta_cost > 0).all(), "추가 비용이 0 이하인 구간이 있다")
        self.assertTrue((seg.delta_gain > 0).all(), "이득이 0 이하인 구간이 있다")

    def test_roi_decreases_within_an_episode(self) -> None:
        """한 문항 안에서 ROI가 감소해야 전역 정렬이 순서를 지켜 준다."""

        s, c, keys = synthetic(600, 3)
        seg = build_segments(s, c, keys)
        by_ep: dict[int, list[float]] = {}
        for k in range(len(seg.episode)):
            by_ep.setdefault(int(seg.episode[k]), []).append(float(seg.roi[k]))
        multi = [v for v in by_ep.values() if len(v) > 1]
        self.assertGreater(len(multi), 0, "2단 승격이 있는 문항이 있어야 의미 있는 검사다")
        for chain in multi:
            self.assertEqual(chain, sorted(chain, reverse=True))

    def test_allow_mask_is_respected(self) -> None:
        s, c, keys = synthetic(500, 4)
        allow = np.ones((len(s), 3), dtype=bool)
        allow[:, 2] = False  # K1 전면 차단
        picks = allocate(
            s, c, multiplier=4.0, util=1.0, allow=allow, keys=keys
        ).picks
        self.assertEqual(0, int((picks == 2).sum()), "차단한 K1이 선택됐다")

    def test_light_is_always_available(self) -> None:
        """allow가 전부 False여도 light로 떨어져야 한다. 폴백이 없으면 안 된다."""

        s, c, keys = synthetic(50, 5)
        allow = np.zeros((len(s), 3), dtype=bool)
        picks = allocate(s, c, multiplier=4.0, util=1.0, allow=allow, keys=keys).picks
        self.assertTrue(set(np.unique(picks)) <= {0}, "light 외의 모델이 선택됐다")


class BudgetTest(unittest.TestCase):
    def test_never_exceeds_target(self) -> None:
        for seed in range(6):
            s, c, keys = synthetic(700, seed)
            light = math.fsum(c[:, 0].tolist())
            for multiplier in (1.25, 2.0, 4.0):
                for util in (0.80, 0.90, 1.00):
                    picks = allocate(
                        s, c, multiplier=multiplier, util=util, keys=keys
                    ).picks
                    used = math.fsum(c[np.arange(len(s)), picks].tolist())
                    self.assertLessEqual(
                        used,
                        light * multiplier * util + 1e-12,
                        f"seed={seed} mult={multiplier} util={util}: 목표 초과",
                    )

    def test_zero_budget_stays_at_base(self) -> None:
        s, c, keys = synthetic(200, 7)
        picks = allocate(s, c, multiplier=1.0, util=0.0, keys=keys).picks
        np.testing.assert_array_equal(picks, np.argmin(c, axis=1))

    def test_generous_budget_reaches_unconstrained_optimum(self) -> None:
        s, c, keys = synthetic(300, 8)
        picks = allocate(s, c, multiplier=1e6, util=1.0, keys=keys).picks
        np.testing.assert_array_equal(picks, np.argmax(s, axis=1))

    def test_matches_bisection_closely(self) -> None:
        """독립 구현인 λ 이분탐색과 점수가 어긋나지 않아야 한다."""

        for seed in (10, 11, 12):
            s, c, keys = synthetic(800, seed)
            for multiplier in (1.25, 2.0, 4.0):
                g = allocate(s, c, multiplier=multiplier, util=0.9, keys=keys).picks
                b = allocate_bisect(s, c, multiplier=multiplier, util=0.9).picks
                a = np.arange(len(s))
                self.assertGreaterEqual(
                    s[a, g].sum(),
                    s[a, b].sum() - 1e-9,
                    f"seed={seed} mult={multiplier}: 그리디가 이분탐색보다 나쁘다",
                )


class GroupAtomicityTest(unittest.TestCase):
    def test_identical_prompts_are_promoted_together(self) -> None:
        """같은 프롬프트는 예산이 빠듯해도 함께 올라가거나 함께 남아야 한다.

        하나만 올라가면 '같은 프롬프트와 등급의 선택은 같아야 한다'를 어긴다.
        """

        rng = np.random.default_rng(21)
        for trial in range(40):
            n = 120
            s, c, _ = synthetic(n, 100 + trial)
            # 각 문항을 두 번씩 넣어 완전한 중복 쌍을 만든다.
            s2 = np.repeat(s, 2, axis=0)
            c2 = np.repeat(c, 2, axis=0)
            keys2 = [content_key(f"dup{i}") for i in range(n) for _ in range(2)]
            multiplier = float(rng.choice([1.25, 2.0, 4.0]))
            util = float(rng.uniform(0.3, 1.0))
            picks = allocate(
                s2, c2, multiplier=multiplier, util=util, keys=keys2
            ).picks
            np.testing.assert_array_equal(
                picks[0::2],
                picks[1::2],
                err_msg=f"trial={trial} mult={multiplier} util={util:.2f}: 중복 쌍의 선택이 갈렸다",
            )

    def test_group_atomicity_holds_without_explicit_keys(self) -> None:
        """keys를 안 넘겨도 예측값에서 키를 만들어 같은 그룹으로 묶어야 한다."""

        s, c, _ = synthetic(150, 33)
        s2 = np.repeat(s, 3, axis=0)
        c2 = np.repeat(c, 3, axis=0)
        picks = allocate(s2, c2, multiplier=2.0, util=0.55).picks
        np.testing.assert_array_equal(picks[0::3], picks[1::3])
        np.testing.assert_array_equal(picks[0::3], picks[2::3])


class OrderInvarianceTest(unittest.TestCase):
    def test_greedy_is_order_invariant(self) -> None:
        rng = np.random.default_rng(41)
        for seed in range(4):
            s, c, keys = synthetic(900, 200 + seed)
            for multiplier, util in ((1.25, 0.90), (2.0, 0.88), (4.0, 0.85)):
                base = allocate(
                    s, c, multiplier=multiplier, util=util, keys=keys
                ).picks
                for _ in range(6):
                    o = rng.permutation(len(s))
                    moved = allocate(
                        s[o],
                        c[o],
                        multiplier=multiplier,
                        util=util,
                        keys=[keys[i] for i in o],
                    ).picks
                    restored = np.empty_like(moved)
                    restored[o] = moved
                    np.testing.assert_array_equal(
                        restored,
                        base,
                        err_msg=f"seed={seed} mult={multiplier}: 순서가 선택을 바꿨다",
                    )


if __name__ == "__main__":
    unittest.main()
