# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""설정 파이프라인, 레지스트리, 캐시, 챔피언 승격 규칙."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from router.cache import ArrayCache, digest, texts_digest  # noqa: E402
from router.data import TIERS, combine_datasets, load_dataset  # noqa: E402
from router.heads import (  # noqa: E402
    COST_HEADS,
    GATES,
    SCORE_HEADS,
    _hash_ridge_oof_apply,
    build_cost_head,
    build_gate,
    build_score_head,
)
from router.pipeline import Config, predict, run_cv, run_on_split  # noqa: E402

_DATA = {}


def datasets():
    if not _DATA:
        _DATA["train"] = load_dataset("train")
        _DATA["dev"] = load_dataset("dev")
    return _DATA["train"], _DATA["dev"]


class CombinedDatasetTest(unittest.TestCase):
    def test_public_train_dev_combination_preserves_rows(self) -> None:
        train, dev = datasets()
        combined = combine_datasets(train, dev)
        self.assertEqual(2640, len(combined))
        self.assertEqual("public-train-dev", combined.split)
        self.assertEqual(train.episode_ids + dev.episode_ids, combined.episode_ids)
        np.testing.assert_array_equal(train.score, combined.score[: len(train)])
        np.testing.assert_array_equal(dev.cost, combined.cost[len(train) :])
        self.assertEqual(combined.split, combined.inputs.split)
        self.assertEqual(combined.split, combined.outcomes.split)

    def test_combination_rejects_duplicate_episodes(self) -> None:
        train, _ = datasets()
        with self.assertRaisesRegex(ValueError, "중복"):
            combine_datasets(train, train)


class ConfigTest(unittest.TestCase):
    def test_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({"id": "x", "typo": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                Config.load(path)

    def test_util_accepts_scalar_and_mapping(self) -> None:
        self.assertEqual(
            {t: 0.8 for t in TIERS}, Config(id="a", alloc={"util": 0.8}).util
        )
        merged = Config(id="a", alloc={"util": {"premium": 0.7}}).util
        self.assertEqual(0.7, merged["premium"])
        self.assertEqual(0.90, merged["fast"])  # 지정 안 한 등급은 기본값

    def test_defaults_are_conservative(self) -> None:
        """기본 사용률이 1.0이면 안 된다. 실수로 예산을 꽉 채우게 된다."""

        for value in Config(id="a").util.values():
            self.assertLess(value, 1.0)

    def test_size_penalty_shrinks_small_batches_more(self) -> None:
        """실현 비율의 흔들림은 표본 수의 제곱근에 반비례한다.

        200문항 배치에서 파산이 몰렸다. 작은 배치일수록 마진을 더 줘야 한다.
        """

        from router.pipeline import effective_util

        M = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(id="a", alloc={"util": 0.90, "size_penalty": 2.0})
        small = effective_util(config, 200, M)["fast"]
        medium = effective_util(config, 880, M)["fast"]
        large = effective_util(config, 2640, M)["fast"]
        self.assertLess(small, medium)
        self.assertLess(medium, large)
        self.assertLess(large, 0.90)
        self.assertAlmostEqual(0.90 - 2.0 / 200 ** 0.5, small, places=9)

    def test_size_penalty_defaults_to_off(self) -> None:
        from router.pipeline import effective_util

        M = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(id="a", alloc={"util": 0.9})
        self.assertEqual(config.util, effective_util(config, 100, M))

    def test_headroom_is_comparable_across_tiers(self) -> None:
        """headroom=h면 세 등급이 모두 여윳돈의 h를 쓴다.

        util은 전체 예산 대비 비율이라 같은 값이 등급마다 다른 뜻이 된다.
        util 0.9면 Fast는 여윳돈의 50%, Premium은 87%를 쓴다. 그 비대칭
        때문에 util에 일률적으로 마진을 빼면 Fast만 통째로 죽는다.
        """

        from router.pipeline import effective_util

        M = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(id="a", alloc={"headroom": 0.8})
        util = effective_util(config, 1000, M)
        for tier, multiplier in M.items():
            used = util[tier] * multiplier
            self.assertAlmostEqual(1.0 + 0.8 * (multiplier - 1.0), used, places=9)

    def test_headroom_takes_priority_over_util(self) -> None:
        from router.pipeline import effective_util

        M = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(id="a", alloc={"util": 0.5, "headroom": 0.9})
        self.assertAlmostEqual(
            1.0 + 0.9 * 0.25, effective_util(config, 1000, M)["fast"] * 1.25, places=9
        )

    def test_size_penalty_never_goes_negative(self) -> None:
        from router.pipeline import effective_util

        M = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(id="a", alloc={"util": 0.5, "size_penalty": 50.0})
        for value in effective_util(config, 4, M).values():
            self.assertGreaterEqual(value, 0.0)

    def test_mu_accepts_tier_specific_mapping(self) -> None:
        config = Config(id="a", alloc={"mu": {"fast": 1.0, "premium": 0.25}})
        self.assertEqual(1.0, config.mu_for_tier("fast"))
        self.assertEqual(0.0, config.mu_for_tier("balanced"))
        self.assertEqual(0.25, config.mu_for_tier("premium"))

    def test_size_penalty_accepts_tier_specific_mapping(self) -> None:
        from router.pipeline import effective_util

        multipliers = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
        config = Config(
            id="a",
            alloc={
                "headroom": 0.5,
                "size_penalty": {"fast": 2.2, "balanced": 1.0},
            },
        )
        util = effective_util(config, 100, multipliers)
        fast_used = util["fast"] * multipliers["fast"]
        balanced_used = util["balanced"] * multipliers["balanced"]
        premium_used = util["premium"] * multipliers["premium"]
        self.assertAlmostEqual(1 + (0.5 - 0.22) * 0.25, fast_used)
        self.assertAlmostEqual(1 + (0.5 - 0.10) * 1.0, balanced_used)
        self.assertAlmostEqual(1 + 0.5 * 3.0, premium_used)

    def test_relative_cost_cap_accepts_scalar_and_tier_mapping(self) -> None:
        scalar = Config(id="a", alloc={"relative_cost_cap": 8.0})
        self.assertEqual(8.0, scalar.relative_cost_cap_for_tier("premium"))
        mapped = Config(
            id="b", alloc={"relative_cost_cap": {"fast": 4.0, "premium": 30.0}}
        )
        self.assertEqual(4.0, mapped.relative_cost_cap_for_tier("fast"))
        self.assertTrue(np.isinf(mapped.relative_cost_cap_for_tier("balanced")))
        with self.assertRaisesRegex(ValueError, "1보다 커야"):
            Config(id="c", alloc={"relative_cost_cap": 1.0}).relative_cost_cap


class RegistryTest(unittest.TestCase):
    def test_unknown_name_is_rejected_with_options(self) -> None:
        for builder, table in (
            (build_score_head, SCORE_HEADS),
            (build_cost_head, COST_HEADS),
            (build_gate, GATES),
        ):
            with self.assertRaises(KeyError) as ctx:
                builder("존재하지-않는-구현")
            self.assertIn(sorted(table)[0], str(ctx.exception))

    def test_kwargs_reach_the_implementation(self) -> None:
        head = build_cost_head({"name": "family", "quantile": 0.9})
        self.assertEqual(0.9, head.quantile)
        self.assertIn("0.9", head.version)

    def test_versions_change_with_hyperparameters(self) -> None:
        a = build_cost_head({"name": "family", "quantile": 0.5}).version
        b = build_cost_head({"name": "family", "quantile": 0.8}).version
        self.assertNotEqual(a, b, "설정이 다른데 버전 문자열이 같으면 캐시가 오염된다")


class HeadContractTest(unittest.TestCase):
    def test_heads_take_only_text(self) -> None:
        """계약이 프롬프트만 받는지 시그니처로 확인한다 (RULES A1)."""

        import inspect

        for builder, name in (
            (build_score_head, "family"),
            (build_cost_head, "family"),
        ):
            params = list(
                inspect.signature(builder(name).predict).parameters
            )
            self.assertEqual(["texts"], params, f"{name}: 예측이 텍스트 외의 값을 받는다")

    def test_shapes_and_ranges(self) -> None:
        train, dev = datasets()
        prediction = predict(Config(id="t", cost={"name": "family", "quantile": 0.75}), train, dev.texts)
        n = len(dev)
        self.assertEqual((n, 3), prediction.s_hat.shape)
        self.assertEqual((n, 3), prediction.c_hat.shape)
        self.assertEqual((n, 3), prediction.allow.shape)
        self.assertTrue((prediction.c_hat > 0).all(), "비용 예측에 0 이하가 있다")
        self.assertTrue(
            ((prediction.s_hat >= 0) & (prediction.s_hat <= 1)).all(),
            "점수 예측이 [0,1] 밖으로 나갔다",
        )

    def test_higher_quantile_predicts_higher_cost(self) -> None:
        """비용 편향은 항상 위쪽이어야 한다 (RULES C2)."""

        train, dev = datasets()
        low = predict(Config(id="l", cost={"name": "family", "quantile": 0.5}), train, dev.texts)
        high = predict(Config(id="h", cost={"name": "family", "quantile": 0.9}), train, dev.texts)
        self.assertTrue((high.c_hat >= low.c_hat - 1e-12).all())
        self.assertGreater(high.c_hat.sum(), low.c_hat.sum())

    def test_hash_cost_uses_worst_risk_for_unseen_family(self) -> None:
        """LOFO에서는 전역 평균 대신 관측 계열 중 최악의 q90으로 폴백한다."""

        from router.features import FAMILY_INDEX, family_codes

        train, _dev = datasets()
        codes = family_codes(train.texts)
        held_out = FAMILY_INDEX["other"]
        fit_part = train.subset(np.where(codes != held_out)[0])
        texts = tuple(
            train.texts[i] for i in np.where(codes == held_out)[0][:8]
        )
        base = build_cost_head(
            {"name": "hash_ridge", "risk_quantile": 0.9}
        )
        guarded = build_cost_head(
            {
                "name": "hash_ridge",
                "risk_quantile": 0.9,
                "unseen_family_risk": True,
            }
        )
        base.fit(fit_part)
        guarded.fit(fit_part)
        base_cost, _ = base.predict(texts)
        guarded_cost, _ = guarded.predict(texts)
        self.assertTrue((guarded_cost[:, 1:] >= base_cost[:, 1:] - 1e-12).all())
        self.assertGreater(guarded_cost[:, 1:].sum(), base_cost[:, 1:].sum())

        revived = build_cost_head(
            {
                "name": "hash_ridge",
                "risk_quantile": 0.9,
                "unseen_family_risk": True,
            }
        )
        revived.load_state(
            json.loads(json.dumps(guarded.state())), train.policy
        )
        revived_cost, _ = revived.predict(texts)
        np.testing.assert_allclose(guarded_cost, revived_cost)

    def test_hash_cost_oof_risk_does_not_fit_its_own_row(self) -> None:
        """검증 행의 target을 바꿔도 그 행의 OOF 예측은 변하면 안 된다."""

        design = np.arange(24, dtype=float).reshape(8, 3)
        target = np.stack(
            [np.linspace(0.0, 1.0, 8), np.linspace(1.0, 3.0, 8)], axis=1
        )
        folds = [np.array([0, 2, 4, 6]), np.array([1, 3, 5, 7])]
        before = _hash_ridge_oof_apply(design, target, folds, alpha=10.0)
        changed = target.copy()
        changed[0] += 1_000_000.0
        after = _hash_ridge_oof_apply(design, changed, folds, alpha=10.0)
        np.testing.assert_allclose(before[0], after[0], rtol=0.0, atol=0.0)
        self.assertFalse(np.allclose(before[1], after[1]))

    def test_hash_cost_oof_risk_is_order_invariant_and_round_trips(self) -> None:
        """content-hash fold 위험 추정은 입력 순서와 직렬화에 무관해야 한다."""

        train, dev = datasets()
        spec = {
            "name": "hash_ridge",
            "risk_quantile": 0.8,
            "unseen_family_risk": True,
            "risk_oof_folds": 4,
            "conditional_risk_families": ["code_io"],
            "conditional_risk_alpha": 10,
            "conditional_risk_bins": 128,
            "conditional_risk_strength": 2,
        }
        original = build_cost_head(spec)
        reversed_head = build_cost_head(spec)
        original.fit(train)
        reversed_head.fit(train.subset(np.arange(len(train))[::-1]))
        texts = dev.texts[:16]
        original_cost, _ = original.predict(texts)
        reversed_cost, _ = reversed_head.predict(texts)
        np.testing.assert_allclose(original_cost, reversed_cost, rtol=1e-10)

        revived = build_cost_head(spec)
        revived.load_state(json.loads(json.dumps(original.state())), train.policy)
        revived_cost, _ = revived.predict(texts)
        np.testing.assert_allclose(original_cost, revived_cost)

    def test_hash_cost_conditional_risk_only_inflates_active_family(self) -> None:
        from router.features import FAMILY_INDEX, family_codes

        train, dev = datasets()
        base_spec = {
            "name": "hash_ridge",
            "risk_quantile": 0.8,
            "unseen_family_risk": True,
            "risk_oof_folds": 4,
        }
        guarded_spec = {
            **base_spec,
            "conditional_risk_families": ["code_io"],
            "conditional_risk_alpha": 10,
            "conditional_risk_bins": 128,
            "conditional_risk_strength": 2,
        }
        base = build_cost_head(base_spec)
        guarded = build_cost_head(guarded_spec)
        base.fit(train)
        guarded.fit(train)
        base_cost, _ = base.predict(dev.texts)
        guarded_cost, _ = guarded.predict(dev.texts)
        active = family_codes(dev.texts) == FAMILY_INDEX["code_io"]
        np.testing.assert_allclose(guarded_cost[~active], base_cost[~active])
        self.assertTrue((guarded_cost[active, 1:] >= base_cost[active, 1:]).all())
        self.assertGreater(
            float((guarded_cost[active, 1:] - base_cost[active, 1:]).sum()),
            0.0,
        )

    def test_hash_cost_oof_risk_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "2 이상"):
            build_cost_head(
                {"name": "hash_ridge", "risk_quantile": 0.9, "risk_oof_folds": 1}
            )
        with self.assertRaisesRegex(ValueError, "risk_quantile"):
            build_cost_head({"name": "hash_ridge", "risk_oof_folds": 4})
        with self.assertRaisesRegex(ValueError, "risk_oof_folds"):
            build_cost_head(
                {
                    "name": "hash_ridge",
                    "risk_quantile": 0.8,
                    "conditional_risk_families": ["code_io"],
                }
            )

        design = np.arange(12, dtype=float).reshape(4, 3)
        target = np.arange(4, dtype=float)
        with self.assertRaisesRegex(ValueError, "정확히 포함"):
            _hash_ridge_oof_apply(
                design,
                target,
                [np.array([0, 1]), np.array([1, 2, 3])],
                alpha=10.0,
            )

    def test_tiered_cost_uses_independent_heads_and_round_trips(self) -> None:
        train, dev = datasets()
        spec = {
            "name": "tiered",
            "heads": {
                "fast": {"name": "family", "quantile": 0.5},
                "balanced": {"name": "family", "quantile": 0.75},
                "premium": {"name": "family", "quantile": 0.9},
            },
        }
        head = build_cost_head(spec)
        head.fit(train)
        fast, _ = head.predict_tier(dev.texts[:40], "fast")
        premium, _ = head.predict_tier(dev.texts[:40], "premium")
        self.assertGreater(float((premium - fast).sum()), 0.0)
        np.testing.assert_allclose(head.predict(dev.texts[:40])[0], fast)

        revived = build_cost_head(spec)
        revived.load_state(json.loads(json.dumps(head.state())), train.policy)
        for tier in TIERS:
            expected = head.predict_tier(dev.texts[:40], tier)[0]
            actual = revived.predict_tier(dev.texts[:40], tier)[0]
            np.testing.assert_allclose(expected, actual)

        prediction = predict(Config(id="tiered-cost", cost=spec), train, dev.texts[:40])
        self.assertIsNotNone(prediction.c_hat_by_tier)
        np.testing.assert_allclose(prediction.cost_for_tier("fast"), fast)

    def test_tiered_cost_requires_all_official_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "등급 오류"):
            build_cost_head(
                {"name": "tiered", "heads": {"fast": "family"}}
            )

    def test_roi_gate_blocks_the_known_bad_families(self) -> None:
        train, dev = datasets()
        prediction = predict(Config(id="g", gate={"name": "family_roi", "min_roi": 1.0}), train, dev.texts)
        from router.features import FAMILY_INDEX, family_codes

        codes = family_codes(dev.texts)
        for name in ("logic", "long_ctx", "ko_open"):
            m = codes == FAMILY_INDEX[name]
            if m.any():
                self.assertFalse(
                    prediction.allow[m, 2].any(), f"{name} 계열의 K1이 열려 있다"
                )
        self.assertTrue(prediction.allow[:, 0].all(), "light가 막힌 문항이 있다")

    def test_tail_exposure_gate_is_tier_specific_and_round_trips(self) -> None:
        train, _dev = datasets()
        spec = {
            "name": "tail_exposure",
            "fast_code_ax31_cap": 3.25,
            "balanced_code_ax31_cap": 3.5,
            "premium_code_k1_cap": 200.0,
            "block_code_k1": True,
            "block_other_k1": True,
        }
        texts = (
            "What is 999999999999999999999999999999999999999999?",
            r"Let \\frac{x}{((y+1))}=3 and solve exactly.",
            "def f(text):\n    return text.replace('a', 'b')\nassert f('a') == 'b'",
            "If someone is kind, is it true or false?",
            "ordinary short question",
        )
        score = np.full((len(texts), 3), 0.5)
        cost = np.asarray(
            [[1, 12, 300], [1, 3, 40], [1, 4, 250], [1, 2, 3], [1, 2, 3]],
            dtype=float,
        )
        gate = build_gate(spec)
        gate.fit(train)
        fast = gate.allow_tier(texts, score, cost, "fast")
        balanced = gate.allow_tier(texts, score, cost, "balanced")
        premium = gate.allow_tier(texts, score, cost, "premium")
        self.assertFalse(fast[0, 1])
        self.assertFalse(balanced[0, 1])
        self.assertFalse(premium[1, 2])
        self.assertFalse(fast[2, 1])
        self.assertFalse(balanced[2, 1])
        self.assertFalse(premium[2, 2])
        self.assertFalse(fast[2, 2])
        self.assertFalse(premium[4, 2])
        self.assertTrue(fast[:, 0].all())
        self.assertTrue(premium[3].all())

        revived = build_gate(spec)
        revived.load_state(json.loads(json.dumps(gate.state())))
        np.testing.assert_array_equal(
            premium, revived.allow_tier(texts, score, cost, "premium")
        )


class FamilyMixtureTest(unittest.TestCase):
    """T6 — 확률 이중계산을 고친 혼합 헤드.

    ``family_useful``은 문항별 이득을 ``계열평균이득 × P(유용)``으로 봤는데
    계열평균이득에 이미 계열의 평균 유용확률이 들어 있어 확률이 두 번
    곱해졌다. 예측 이득 평균이 0.146에서 0.061로 찌그러졌다.
    """

    def test_strength_zero_is_exactly_the_family_head(self) -> None:
        """혼합은 계열 평균의 **엄밀한 일반화**여야 한다.

        전체 기댓값의 법칙에 의해 ``P·E[·|유용] + (1−P)·E[·|무용]``은 계열
        평균과 정확히 같다. 여기가 어긋나면 혼합 자체가 틀린 것이다.
        """

        train, dev = datasets()
        family = build_score_head("family")
        family.fit(train)
        mixture = build_score_head({"name": "family_mixture", "strength": 0.0})
        mixture.fit(train)
        np.testing.assert_allclose(
            family.predict(dev.texts), mixture.predict(dev.texts), atol=1e-9
        )

    def test_keeps_the_gain_level_unlike_family_useful(self) -> None:
        train, dev = datasets()
        gains = {}
        for name in ("family", "family_useful", "family_mixture"):
            head = build_score_head(name)
            head.fit(train)
            p = head.predict(dev.texts)
            gains[name] = float((p[:, 1:] - p[:, [0]]).mean())

        self.assertLess(
            gains["family_useful"], gains["family"] * 0.7,
            "family_useful이 이득을 찌그러뜨리지 않는다면 이 테스트의 전제가 낡았다",
        )
        self.assertGreater(
            gains["family_mixture"], gains["family"] * 0.9,
            "혼합이 이득 수준을 지키지 못한다",
        )

    def test_adds_per_episode_spread(self) -> None:
        """계열 평균은 같은 계열 안에서 모든 문항에 같은 이득을 준다.

        혼합의 존재 이유는 그 안에서 순서를 매기는 것이다.
        """

        train, dev = datasets()
        from router.features import family_codes

        codes = family_codes(dev.texts)
        head = build_score_head("family_mixture")
        head.fit(train)
        p = head.predict(dev.texts)
        gain = p[:, 2] - p[:, 0]

        spread = [
            float(gain[codes == f].std())
            for f in np.unique(codes)
            if (codes == f).sum() >= 20
        ]
        self.assertTrue(spread, "표본이 충분한 계열이 없다")
        self.assertGreater(
            min(spread), 0.0, "어떤 계열 안에서도 문항별 차이가 없다"
        )

    def test_predictions_stay_in_range(self) -> None:
        train, dev = datasets()
        head = build_score_head("family_mixture")
        head.fit(train)
        p = head.predict(dev.texts)
        self.assertTrue(((p >= 0.0) & (p <= 1.0)).all(), "점수가 [0,1] 밖이다")

    def test_can_limit_episode_spread_to_selected_families(self) -> None:
        """T0에서 손실이 집중된 계열만 국소적으로 순서를 바꿀 수 있어야 한다."""

        train, dev = datasets()
        from router.features import FAMILY_INDEX, family_codes

        family = build_score_head("family")
        family.fit(train)
        mixture = build_score_head({
            "name": "family_mixture",
            "strength": 1.0,
            "active_families": ["sym_math", "code_io"],
        })
        mixture.fit(train)
        base = family.predict(dev.texts)
        changed = mixture.predict(dev.texts)
        codes = family_codes(dev.texts)
        active = np.isin(codes, [FAMILY_INDEX["sym_math"], FAMILY_INDEX["code_io"]])
        np.testing.assert_allclose(changed[~active], base[~active], atol=1e-9)
        self.assertGreater(float(np.abs(changed[active] - base[active]).sum()), 0.0)

    def test_rejects_unknown_active_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "알 수 없는 계열"):
            build_score_head({
                "name": "family_mixture",
                "active_families": ["not-a-family"],
            })

    def test_state_round_trip(self) -> None:
        """산출물로 굳혔다 되살려도 같은 답이어야 한다. 어긋나면 런타임이
        조용히 폴백으로 떨어진다."""

        train, dev = datasets()
        head = build_score_head("family_mixture")
        head.fit(train)
        before = head.predict(dev.texts)

        revived = build_score_head("family_mixture")
        revived.load_state(json.loads(json.dumps(head.state())))
        np.testing.assert_allclose(before, revived.predict(dev.texts), atol=1e-12)


class ResponseShapeTest(unittest.TestCase):
    def test_strength_zero_is_exactly_family_score(self) -> None:
        train, dev = datasets()
        family = build_score_head("family")
        family.fit(train)
        response = build_score_head({"name": "response_shape", "strength": 0.0})
        response.fit(train)
        np.testing.assert_allclose(
            response.predict(dev.texts), family.predict(dev.texts), atol=1e-9
        )

    def test_only_changes_selected_families(self) -> None:
        train, dev = datasets()
        from router.features import FAMILY_INDEX, family_codes

        family = build_score_head("family")
        family.fit(train)
        response = build_score_head({
            "name": "response_shape",
            "active_families": ["sym_math", "code_io"],
        })
        response.fit(train)
        base = family.predict(dev.texts)
        changed = response.predict(dev.texts)
        codes = family_codes(dev.texts)
        active = np.isin(codes, [FAMILY_INDEX["sym_math"], FAMILY_INDEX["code_io"]])
        np.testing.assert_allclose(changed[~active], base[~active], atol=1e-9)
        self.assertGreater(float(np.abs(changed[active] - base[active]).sum()), 0.0)

    def test_predictions_stay_in_range(self) -> None:
        train, dev = datasets()
        head = build_score_head("response_shape")
        head.fit(train)
        predicted = head.predict(dev.texts)
        self.assertTrue(((predicted >= 0.0) & (predicted <= 1.0)).all())

    def test_state_round_trip(self) -> None:
        train, dev = datasets()
        head = build_score_head({
            "name": "response_shape",
            "strength": 0.75,
            "active_families": ["sym_math", "code_io"],
        })
        head.fit(train)
        before = head.predict(dev.texts)
        revived = build_score_head({
            "name": "response_shape",
            "strength": 0.75,
            "active_families": ["sym_math", "code_io"],
        })
        revived.load_state(json.loads(json.dumps(head.state())))
        np.testing.assert_allclose(before, revived.predict(dev.texts), atol=1e-12)


class TemplateScoreTest(unittest.TestCase):
    def test_unknown_templates_fall_back_to_family(self) -> None:
        train, _dev = datasets()
        family = build_score_head("family")
        family.fit(train)
        template = build_score_head("template")
        template.fit(train)
        texts = ("completely unseen synthetic prompt 9918273645",)
        np.testing.assert_allclose(
            template.predict(texts), family.predict(texts), atol=1e-12
        )

    def test_strength_zero_is_family_score(self) -> None:
        train, dev = datasets()
        family = build_score_head("family")
        family.fit(train)
        template = build_score_head({"name": "template", "strength": 0.0})
        template.fit(train)
        np.testing.assert_allclose(
            template.predict(dev.texts), family.predict(dev.texts), atol=1e-12
        )

    def test_state_round_trip(self) -> None:
        train, dev = datasets()
        spec = {
            "name": "template",
            "scheme": "identifiers",
            "prior": 2.0,
            "active_families": ["sym_math"],
        }
        head = build_score_head(spec)
        head.fit(train)
        before = head.predict(dev.texts)
        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(head.state())))
        np.testing.assert_allclose(before, revived.predict(dev.texts), atol=1e-12)

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "정규화"):
            build_score_head({"name": "template", "scheme": "unknown"})
        with self.assertRaisesRegex(ValueError, "prior"):
            build_score_head({"name": "template", "prior": -1})


class BlendScoreTest(unittest.TestCase):
    def test_is_exact_weighted_average_and_round_trips(self) -> None:
        train, dev = datasets()
        specs = [
            {
                "name": "family_mixture",
                "strength": 0.75,
                "active_families": ["sym_math", "code_io"],
            },
            {
                "name": "template",
                "scheme": "digits",
                "prior": 1.0,
                "active_families": ["sym_math"],
            },
        ]
        parts = []
        for spec in specs:
            head = build_score_head(spec)
            head.fit(train)
            parts.append(head.predict(dev.texts))

        spec = {"name": "blend", "heads": specs, "weights": [0.8, 0.2]}
        blend = build_score_head(spec)
        blend.fit(train)
        predicted = blend.predict(dev.texts)
        np.testing.assert_allclose(predicted, 0.8 * parts[0] + 0.2 * parts[1])

        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(blend.state())))
        np.testing.assert_allclose(predicted, revived.predict(dev.texts), atol=1e-12)

    def test_rejects_invalid_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "둘 이상"):
            build_score_head({"name": "blend", "heads": ["family"], "weights": [1]})
        with self.assertRaisesRegex(ValueError, "길이"):
            build_score_head({
                "name": "blend",
                "heads": ["family", "global"],
                "weights": [1],
            })
        with self.assertRaisesRegex(ValueError, "합이 양수"):
            build_score_head({
                "name": "blend",
                "heads": ["family", "global"],
                "weights": [0, 0],
            })


class FamilyBlendScoreTest(unittest.TestCase):
    def test_only_active_families_change_and_state_round_trips(self) -> None:
        from router.features import FAMILY_INDEX, family_codes

        train, dev = datasets()
        base_spec = {"name": "hash_ridge", "alpha": 32000, "bins": 256}
        challenger_spec = {
            "name": "response_shape",
            "strength": 0.75,
            "active_families": ["sym_math", "code_io"],
        }
        spec = {
            "name": "family_blend",
            "base": base_spec,
            "challenger": challenger_spec,
            "weight": 0.25,
            "active_families": ["sym_math", "code_io"],
        }
        base = build_score_head(base_spec)
        base.fit(train)
        base_prediction = base.predict(dev.texts)
        blend = build_score_head(spec)
        blend.fit(train)
        prediction = blend.predict(dev.texts)
        codes = family_codes(dev.texts)
        active = np.isin(
            codes, [FAMILY_INDEX["sym_math"], FAMILY_INDEX["code_io"]]
        )
        np.testing.assert_array_equal(prediction[~active], base_prediction[~active])
        self.assertGreater(float(np.abs(prediction[active] - base_prediction[active]).sum()), 0.0)

        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(blend.state())))
        np.testing.assert_allclose(prediction, revived.predict(dev.texts), atol=1e-12)

    def test_rejects_invalid_scope_and_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "비어"):
            build_score_head({
                "name": "family_blend",
                "base": "family",
                "challenger": "global",
                "weight": 0.2,
                "active_families": [],
            })
        with self.assertRaisesRegex(ValueError, "1 이하"):
            build_score_head({
                "name": "family_blend",
                "base": "family",
                "challenger": "global",
                "weight": 1.1,
                "active_families": ["sym_math"],
            })


class FamilyHashRidgeScoreTest(unittest.TestCase):
    def test_local_models_change_active_families_and_round_trip(self) -> None:
        from router.features import FAMILY_INDEX, family_codes

        train, dev = datasets()
        local_spec = {
            "name": "family_hash_ridge",
            "alpha": 1000,
            "bins": 64,
            "active_families": ["sym_math", "code_io"],
        }
        global_head = build_score_head(
            {"name": "hash_ridge", "alpha": 1000, "bins": 64}
        )
        local_head = build_score_head(local_spec)
        global_head.fit(train)
        local_head.fit(train)
        global_prediction = global_head.predict(dev.texts)
        local_prediction = local_head.predict(dev.texts)
        codes = family_codes(dev.texts)
        inactive = ~np.isin(
            codes, [FAMILY_INDEX["sym_math"], FAMILY_INDEX["code_io"]]
        )
        np.testing.assert_allclose(
            local_prediction[inactive], global_prediction[inactive], atol=1e-12
        )
        self.assertGreater(
            float(np.abs(local_prediction[~inactive] - global_prediction[~inactive]).sum()),
            0.0,
        )

        revived = build_score_head(local_spec)
        revived.load_state(json.loads(json.dumps(local_head.state())))
        np.testing.assert_allclose(
            local_prediction, revived.predict(dev.texts), atol=1e-12
        )

    def test_requires_nonempty_known_scope(self) -> None:
        with self.assertRaisesRegex(ValueError, "비어"):
            build_score_head({"name": "family_hash_ridge"})
        with self.assertRaisesRegex(ValueError, "알 수 없는"):
            build_score_head({
                "name": "family_hash_ridge",
                "active_families": ["not-a-family"],
            })


class HashResponseScoreTest(unittest.TestCase):
    def test_predictions_and_state_round_trip(self) -> None:
        train, dev = datasets()
        spec = {
            "name": "hash_response",
            "alpha": 1000,
            "bins": 64,
            "strength": 0.75,
            "active_families": ["sym_math", "code_io"],
        }
        head = build_score_head(spec)
        head.fit(train)
        prediction = head.predict(dev.texts[:80])
        self.assertEqual((80, 3), prediction.shape)
        self.assertTrue(((prediction >= 0.0) & (prediction <= 1.0)).all())

        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(head.state())))
        np.testing.assert_allclose(
            prediction, revived.predict(dev.texts[:80]), atol=1e-12
        )

    def test_strength_zero_uses_family_stage_means(self) -> None:
        train, dev = datasets()
        head = build_score_head({
            "name": "hash_response",
            "bins": 64,
            "strength": 0.0,
        })
        head.fit(train)
        prediction = head.predict(dev.texts)
        from router.features import family_codes

        codes = family_codes(dev.texts)
        for code in np.unique(codes):
            mask = codes == code
            early = prediction[mask, 1] - prediction[mask, 0]
            self.assertLess(float(early.max() - early.min()), 1e-12)


class HashKNNScoreTest(unittest.TestCase):
    def test_predictions_are_order_invariant_and_round_trip(self) -> None:
        train, dev = datasets()
        spec = {
            "name": "hash_knn",
            "bins": 64,
            "neighbors": 8,
            "prior": 4,
            "power": 2,
            "same_family": True,
        }
        head = build_score_head(spec)
        head.fit(train)
        texts = dev.texts[:40]
        prediction = head.predict(texts)
        reversed_prediction = head.predict(tuple(reversed(texts)))[::-1]
        np.testing.assert_allclose(prediction, reversed_prediction, atol=1e-12)
        self.assertTrue(((prediction >= 0.0) & (prediction <= 1.0)).all())

        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(head.state())))
        np.testing.assert_allclose(
            prediction, revived.predict(texts), atol=1e-12
        )

    def test_large_threshold_falls_back_to_base(self) -> None:
        train, dev = datasets()
        base = build_score_head({"name": "hash_ridge", "alpha": 32000, "bins": 64})
        knn = build_score_head({
            "name": "hash_knn",
            "bins": 64,
            "neighbors": 8,
            "prior": 4,
            "threshold": 0.999999,
        })
        base.fit(train)
        knn.fit(train)
        np.testing.assert_allclose(
            knn.predict(dev.texts[:40]), base.predict(dev.texts[:40]), atol=1e-12
        )

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "neighbors"):
            build_score_head({"name": "hash_knn", "neighbors": 0})
        with self.assertRaisesRegex(ValueError, "prior"):
            build_score_head({"name": "hash_knn", "prior": -1})


class TieredScoreTest(unittest.TestCase):
    def test_uses_independent_heads_and_round_trips(self) -> None:
        train, dev = datasets()
        spec = {
            "name": "tiered",
            "heads": {
                "fast": {"name": "hash_ridge", "alpha": 100, "bins": 256},
                "balanced": "family",
                "premium": "family_mixture",
            },
        }
        head = build_score_head(spec)
        head.fit(train)
        fast = head.predict_tier(dev.texts[:80], "fast")
        balanced = head.predict_tier(dev.texts[:80], "balanced")
        self.assertGreater(float(np.abs(fast - balanced).sum()), 0.0)

        revived = build_score_head(spec)
        revived.load_state(json.loads(json.dumps(head.state())))
        for tier in TIERS:
            np.testing.assert_allclose(
                head.predict_tier(dev.texts[:80], tier),
                revived.predict_tier(dev.texts[:80], tier),
                atol=1e-12,
            )

    def test_requires_exactly_the_official_tiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "등급 오류"):
            build_score_head({"name": "tiered", "heads": {"fast": "family"}})


class PipelineTest(unittest.TestCase):
    def test_reproduces_the_documented_a1_row(self) -> None:
        """GOAL.md에 적힌 A1 수치를 그대로 재현해야 한다.

        문서의 숫자와 코드가 갈리면 문서를 못 믿게 된다. 여기가 그 접점이다.
        A1을 의도적으로 바꿨다면 이 테스트와 GOAL.md를 함께 고친다.
        """

        train, dev = datasets()
        config = Config(id="a1", alloc={"util": 0.90})
        evaluation, _, _ = run_on_split(config, train, dev)
        for tier, expected in (
            ("fast", 0.6511),
            ("balanced", 0.6909),
            ("premium", 0.7327),
        ):
            self.assertAlmostEqual(
                expected, float(evaluation.tiers[tier].score), places=3, msg=tier
            )
        self.assertAlmostEqual(0.6875, float(evaluation.final_score), places=3)

    def test_cv_does_not_leak_the_test_fold(self) -> None:
        """fold를 하나만 남기고 나머지로 적합했는지 확인한다.

        누수가 있으면 CV 점수가 Dev보다 크게 높아진다. 여기서는 fold 적합에
        쓰인 자료에 테스트 fold 문항이 없다는 것을 직접 본다.
        """

        train, _ = datasets()
        folds = train.folds(5)
        self.assertEqual(len(train), sum(len(f) for f in folds))
        all_idx = np.concatenate(folds)
        self.assertEqual(len(train), len(np.unique(all_idx)), "fold가 겹친다")
        for f, test_idx in enumerate(folds):
            fit_idx = np.concatenate([folds[g] for g in range(5) if g != f])
            self.assertEqual(0, len(np.intersect1d(fit_idx, test_idx)))

    def test_folds_are_content_addressed(self) -> None:
        """순서를 바꿔도 같은 fold 배정이 나와야 재현이 성립한다."""

        train, _ = datasets()
        folds = train.folds(5)
        assign = np.zeros(len(train), dtype=int)
        for f, idx in enumerate(folds):
            assign[idx] = f
        rng = np.random.default_rng(0)
        order = rng.permutation(len(train))
        shuffled = train.subset(order)
        moved = np.zeros(len(train), dtype=int)
        for f, idx in enumerate(shuffled.folds(5)):
            moved[idx] = f
        np.testing.assert_array_equal(assign[order], moved)

    def test_cv_is_deterministic(self) -> None:
        train, _ = datasets()
        config = Config(id="d", alloc={"util": 0.85})
        first = run_cv(config, train, k=5)
        second = run_cv(config, train, k=5)
        self.assertEqual(first.final_score, second.final_score)


class CacheTest(unittest.TestCase):
    def test_hit_after_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArrayCache(Path(tmp))
            calls = []

            def compute():
                calls.append(1)
                return {"a": np.arange(5.0)}

            first = cache.get_or_compute("ns", "k", compute)
            second = cache.get_or_compute("ns", "k", compute)
            self.assertEqual(1, len(calls), "캐시가 안 먹었다")
            np.testing.assert_array_equal(first["a"], second["a"])
            self.assertEqual(1, cache.hits)

    def test_disabled_cache_always_recomputes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArrayCache(Path(tmp), enabled=False)
            calls = []
            for _ in range(3):
                cache.get_or_compute("ns", "k", lambda: (calls.append(1), {"a": np.zeros(1)})[1])
            self.assertEqual(3, len(calls))

    def test_digest_is_content_addressed(self) -> None:
        self.assertEqual(digest("a", 1), digest("a", 1))
        self.assertNotEqual(digest("a", 1), digest("a", 2))
        self.assertNotEqual(texts_digest(["x", "y"]), texts_digest(["y", "x"]))


class ChampionTest(unittest.TestCase):
    def _record(
        self, cv: str, gate: bool, cv_passed: bool = True, trials: int = 2000
    ) -> dict:
        return {
            "id": "r",
            "cv": {"final_score": cv, "all_passed": cv_passed},
            "gate": {"passed": gate, "trials": trials},
        }

    def test_promotion_requires_enough_gate_trials(self) -> None:
        """적은 시행에서의 0회는 통과가 아니다 (RULES C4).

        실제로 150회 게이트로 승격된 챔피언이 있었다. rule of three로
        0/150은 실제 파산확률 95% 상한이 2%다.
        """

        import run as runner

        with tempfile.TemporaryDirectory() as tmp:
            original = runner.CHAMPION
            runner.CHAMPION = Path(tmp) / "champion.json"
            try:
                self.assertFalse(
                    runner.maybe_promote(self._record("0.9", gate=True, trials=150))
                )
                self.assertFalse(
                    runner.maybe_promote(
                        self._record("0.9", gate=True, trials=runner.MIN_GATE_TRIALS - 1)
                    )
                )
                self.assertTrue(
                    runner.maybe_promote(
                        self._record("0.9", gate=True, trials=runner.MIN_GATE_TRIALS)
                    )
                )
            finally:
                runner.CHAMPION = original

    def test_promotion_requires_gate_and_cv_budget(self) -> None:
        import run as runner

        with tempfile.TemporaryDirectory() as tmp:
            original = runner.CHAMPION
            runner.CHAMPION = Path(tmp) / "champion.json"
            try:
                self.assertFalse(
                    runner.maybe_promote(self._record("0.9", gate=False)),
                    "게이트 불통과인데 승격됐다",
                )
                self.assertFalse(
                    runner.maybe_promote(self._record("0.9", gate=True, cv_passed=False)),
                    "CV에서 예산을 넘겼는데 승격됐다",
                )
                self.assertTrue(runner.maybe_promote(self._record("0.70", gate=True)))
                self.assertFalse(
                    runner.maybe_promote(self._record("0.69", gate=True)),
                    "챔피언보다 낮은데 승격됐다",
                )
                self.assertTrue(runner.maybe_promote(self._record("0.71", gate=True)))
            finally:
                runner.CHAMPION = original


if __name__ == "__main__":
    unittest.main()
