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
from router.data import TIERS, load_dataset  # noqa: E402
from router.heads import (  # noqa: E402
    COST_HEADS,
    GATES,
    SCORE_HEADS,
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
    def _record(self, cv: str, gate: bool, cv_passed: bool = True) -> dict:
        return {
            "id": "r",
            "cv": {"final_score": cv, "all_passed": cv_passed},
            "gate": {"passed": gate},
        }

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
