# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""파산 게이트가 실제로 파산을 잡는지 본다.

게이트가 통과를 남발하면 없느니만 못하다. 그래서 '반드시 실패해야 하는 정책'과
'절대 실패할 수 없는 정책'을 양쪽에서 고정한다.
"""

from __future__ import annotations

import unittest

import numpy as np

from router.data import TIERS, Dataset, cost_from_tokens
from router.stress import (
    family_dominant_sampler,
    family_reweight_sampler,
    gate_passed,
    run_scenario,
    uniform_sampler,
)

MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def toy_dataset(n: int = 300, seed: int = 0) -> tuple[Dataset, np.ndarray]:
    """정책 파일을 그대로 쓰되 자료만 합성한다."""

    from ossp_router.protocol import load_bundled_policy

    rng = np.random.default_rng(seed)
    policy = load_bundled_policy()
    family = rng.integers(0, 4, n)

    tok_in = rng.integers(200, 3000, (n, 3)).astype(float)
    tok_in[:, 1] = tok_in[:, 0]
    tok_in[:, 2] = tok_in[:, 0]
    out_light = rng.lognormal(5.8, 1.0, n)
    tok_out = np.stack(
        [out_light, out_light * rng.lognormal(0.0, 0.3, n), out_light * rng.lognormal(1.8, 1.1, n)],
        axis=1,
    )
    score = np.sort(rng.random((n, 3)), axis=1)
    cost = cost_from_tokens(tok_in, tok_out, policy)

    dataset = Dataset(
        split="toy",
        challenge_id="toy",
        inputs=None,  # 이 테스트는 채점기를 쓰지 않는다
        outcomes=None,
        policy=policy,
        texts=tuple(f"t{i}" for i in range(n)),
        episode_ids=tuple(f"e{i}" for i in range(n)),
        keys=tuple(f"{i:032x}" for i in range(n)),
        score=score,
        input_tokens=tok_in,
        output_tokens=tok_out,
        generations=np.full(n, 2),
        cost=cost,
    )
    return dataset, family


class SamplerTest(unittest.TestCase):
    def test_samplers_return_requested_size(self) -> None:
        _, family = toy_dataset(240, 1)
        rng = np.random.default_rng(0)
        for sampler in (
            uniform_sampler(240, 500),
            family_reweight_sampler(family, 500, 5.0),
            family_dominant_sampler(family, 500, 0.75),
        ):
            self.assertEqual(500, len(sampler(rng)))

    def test_extreme_shift_concentrates_more_than_mild(self) -> None:
        """conc가 작을수록 한 계열 쏠림이 심해야 시나리오 이름값을 한다."""

        _, family = toy_dataset(400, 2)
        rng = np.random.default_rng(3)
        mild = family_reweight_sampler(family, 600, 5.0)
        extreme = family_reweight_sampler(family, 600, 0.7)

        def top_share(sampler) -> float:
            shares = []
            for _ in range(60):
                idx = sampler(rng)
                counts = np.bincount(family[idx], minlength=4)
                shares.append(counts.max() / counts.sum())
            return float(np.mean(shares))

        self.assertGreater(top_share(extreme), top_share(mild) + 0.05)


class GateSensitivityTest(unittest.TestCase):
    def test_all_light_policy_never_fails(self) -> None:
        """전부 light면 비율이 정확히 1.0이라 어떤 등급도 넘길 수 없다."""

        dataset, family = toy_dataset(300, 4)
        s_hat = np.zeros((300, 3))  # 승격 이득 0 -> 아무것도 안 올린다
        result = run_scenario(
            dataset,
            s_hat,
            dataset.cost,
            uniform_sampler(300, 400),
            scenario="all-light",
            util=1.0,
            multipliers=MULT,
            trials=60,
        )
        self.assertTrue(result.passed)
        for tier in TIERS:
            self.assertEqual(0, result.tiers[tier].failures)

    def test_uniform_cost_scaling_changes_nothing(self) -> None:
        """비용 전체에 상수를 곱해도 배분은 그대로여야 한다.

        예산이 '전 문항 light 대비 비율'이므로 균일 스케일은 상쇄된다.
        따라서 비용 헤드가 맞혀야 하는 것은 **절대값이 아니라 모델 간 상대비**다.
        이 성질을 모르면 '비용을 크게 틀렸는데 왜 안 터지지'에서 헤맨다.
        """

        from router.allocate import allocate

        dataset, _ = toy_dataset(300, 51)
        for scale in (0.1, 3.7, 1000.0):
            for tier, mult in MULT.items():
                a = allocate(
                    dataset.score, dataset.cost, multiplier=mult, util=0.9,
                    keys=dataset.keys,
                ).picks
                b = allocate(
                    dataset.score, dataset.cost * scale, multiplier=mult, util=0.9,
                    keys=dataset.keys,
                ).picks
                np.testing.assert_array_equal(
                    a, b, err_msg=f"scale={scale} tier={tier}: 균일 스케일이 배분을 바꿨다"
                )

    def test_gate_catches_a_policy_that_underrates_the_think_model(self) -> None:
        """K1 비용만 10분의 1로 우기는 정책은 반드시 잡혀야 한다.

        이것이 실제 실패 모드다. K1의 출력 토큰은 프롬프트 길이와 무상관이라
        가장 틀리기 쉽고, 틀리면 곧바로 Premium 예산을 넘긴다.
        """

        dataset, family = toy_dataset(300, 5)
        liar = dataset.cost.copy()
        liar[:, 2] *= 0.1
        result = run_scenario(
            dataset,
            dataset.score,
            liar,
            uniform_sampler(300, 400),
            scenario="liar",
            util=0.90,
            multipliers=MULT,
            trials=60,
        )
        self.assertFalse(result.passed, "K1 비용을 속인 정책이 게이트를 통과했다")
        self.assertGreater(result.tiers["premium"].failure_rate, 0.5)

    def test_lower_utilisation_never_increases_failures(self) -> None:
        """사용률을 낮췄는데 파산이 늘면 게이트 계산이 잘못된 것이다."""

        dataset, family = toy_dataset(300, 6)
        c_hat = dataset.cost.copy()
        c_hat[:, 2] *= 0.45  # K1만 과소추정
        s_hat = dataset.score
        previous = None
        for util in (1.00, 0.90, 0.80, 0.70):
            result = run_scenario(
                dataset,
                s_hat,
                c_hat,
                uniform_sampler(300, 400),
                scenario=f"u{util}",
                util=util,
                multipliers=MULT,
                trials=60,
                seed=7,
            )
            total = sum(result.tiers[t].failures for t in TIERS)
            if previous is not None:
                self.assertLessEqual(total, previous, f"util={util}에서 파산이 늘었다")
            previous = total

    def test_same_seed_gives_same_result(self) -> None:
        dataset, family = toy_dataset(200, 8)
        kwargs = dict(
            scenario="det",
            util=0.9,
            multipliers=MULT,
            trials=25,
            seed=42,
        )
        a = run_scenario(
            dataset, dataset.score, dataset.cost * 0.7,
            uniform_sampler(200, 300), **kwargs
        )
        b = run_scenario(
            dataset, dataset.score, dataset.cost * 0.7,
            uniform_sampler(200, 300), **kwargs
        )
        for tier in TIERS:
            self.assertEqual(a.tiers[tier].failures, b.tiers[tier].failures)
            np.testing.assert_array_equal(
                a.tiers[tier].ratio_over_limit, b.tiers[tier].ratio_over_limit
            )

    def test_gate_passed_requires_every_scenario(self) -> None:
        dataset, family = toy_dataset(200, 9)
        clean = run_scenario(
            dataset, np.zeros((200, 3)), dataset.cost,
            uniform_sampler(200, 300), scenario="clean", util=1.0,
            multipliers=MULT, trials=20,
        )
        liar = dataset.cost.copy()
        liar[:, 2] *= 0.05
        dirty = run_scenario(
            dataset, dataset.score, liar,
            uniform_sampler(200, 300), scenario="dirty", util=1.0,
            multipliers=MULT, trials=20,
        )
        self.assertTrue(gate_passed([clean]))
        self.assertFalse(gate_passed([clean, dirty]))


if __name__ == "__main__":
    unittest.main()
