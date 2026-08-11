# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""비용 헤드 — 여기가 틀리면 등급이 통째로 0점이 된다.

두 가지를 못 박는다.

1. **외삽 폭주 금지.** 선형 회귀는 학습 범위 밖 특징에서 로그 예측을 크게
   밀어올리고, expm1을 지나면 토큰 수가 조 단위가 된다. 실제로 한 fold에서
   예측 light 비용이 실제의 463배가 나와 세 등급이 전부 터졌다.
2. **편향은 light 대비 상대적으로.** 예산 한도의 분모가 light 예측 비용이라
   모든 모델을 똑같이 부풀리면 쓸 수 있는 돈까지 같이 늘어난다.
"""

from __future__ import annotations

import unittest

import numpy as np

from router.data import load_dataset
from router.heads import build_cost_head

_TRAIN = {}


def train_set():
    if "d" not in _TRAIN:
        _TRAIN["d"] = load_dataset("train")
    return _TRAIN["d"]


ADVERSARIAL = [
    "",
    "x",
    "9" * 5_000,                 # 거대한 숫자
    "((" * 3_000 + "))" * 3_000,  # 극단적 중첩
    "\\frac" * 2_000,            # LaTeX 폭탄
    "A. " * 4_000,               # 보기처럼 보이는 것 반복
    "가" * 20_000,               # 한글 장문
    "def f(x):\n" * 900,         # 코드처럼 보이는 것 반복
    "\n" * 10_000,               # 빈 줄만
    "0 " * 8_000,                # 숫자 밀도 1
]


class ExtrapolationTest(unittest.TestCase):
    def test_adversarial_inputs_stay_finite_and_bounded(self) -> None:
        train = train_set()
        head = build_cost_head({"name": "ridge", "z": 0.67})
        head.fit(train)
        cost, sd = head.predict(ADVERSARIAL)

        self.assertTrue(np.isfinite(cost).all(), "비용 예측에 inf/nan이 있다")
        self.assertTrue(np.isfinite(sd).all(), "산포 예측에 inf/nan이 있다")
        self.assertTrue((cost > 0).all(), "비용 예측에 0 이하가 있다")

        ceiling = train.cost.max() * 3.0
        self.assertLess(
            float(cost.max()),
            ceiling,
            f"학습 최대 비용의 3배({ceiling:.3f})를 넘는 예측이 나왔다",
        )

    def test_predictions_never_exceed_the_token_clamp(self) -> None:
        train = train_set()
        head = build_cost_head({"name": "ridge", "z": 1.28})
        head.fit(train)
        cost, _ = head.predict(ADVERSARIAL + list(train.texts[:200]))
        # 토큰 상한 2배 + 입력 토큰 상한 2배로 만든 최악 비용
        worst = float(
            (train.input_tokens.max() * 2 * 6.565
             + train.output_tokens.max() * 2 * 26.260) / 1e6
        )
        self.assertLessEqual(float(cost.max()), worst + 1e-9)

    def test_a_bad_fold_does_not_blow_up_the_denominator(self) -> None:
        """CV fold 하나에서 예측 light 비용이 실제와 크게 어긋나면 안 된다."""

        train = train_set()
        folds = train.folds(5)
        for f, test_idx in enumerate(folds):
            fit_idx = np.concatenate([folds[g] for g in range(5) if g != f])
            head = build_cost_head({"name": "ridge", "z": 0.0})
            head.fit(train.subset(fit_idx))
            target = train.subset(test_idx)
            cost, _ = head.predict(target.texts)
            ratio = cost[:, 0].sum() / target.cost[:, 0].sum()
            self.assertLess(
                ratio, 3.0, f"fold{f}: 예측 light 비용이 실제의 {ratio:.1f}배다"
            )
            self.assertGreater(ratio, 0.2, f"fold{f}: 예측 light 비용이 너무 낮다")


class BiasDirectionTest(unittest.TestCase):
    def test_z_raises_upgrade_cost_only(self) -> None:
        """z는 승격 모델만 올리고 light는 건드리지 않아야 한다."""

        train = train_set()
        texts = list(train.texts[:400])
        low = build_cost_head({"name": "ridge", "z": 0.0})
        high = build_cost_head({"name": "ridge", "z": 1.28})
        low.fit(train)
        high.fit(train)
        c_low, _ = low.predict(texts)
        c_high, _ = high.predict(texts)

        np.testing.assert_allclose(
            c_low[:, 0], c_high[:, 0], rtol=1e-9,
            err_msg="z가 light 비용까지 바꿨다. 그러면 예산 분모가 커진다",
        )
        self.assertGreater(c_high[:, 1].sum(), c_low[:, 1].sum())
        self.assertGreater(c_high[:, 2].sum(), c_low[:, 2].sum())

    def test_z_light_lowers_the_denominator(self) -> None:
        """z_light가 음수면 light 예측이 내려가 한도가 줄어야 한다."""

        train = train_set()
        texts = list(train.texts[:400])
        flat = build_cost_head({"name": "ridge", "z": 0.67, "z_light": 0.0})
        down = build_cost_head({"name": "ridge", "z": 0.67, "z_light": -0.5})
        flat.fit(train)
        down.fit(train)
        self.assertLess(
            down.predict(texts)[0][:, 0].sum(), flat.predict(texts)[0][:, 0].sum()
        )

    def test_relative_ratio_grows_with_z(self) -> None:
        """안전의 본질은 절대값이 아니라 K1/light 비율이 커지는 것이다."""

        train = train_set()
        texts = list(train.texts[:400])
        ratios = []
        for z in (0.0, 0.67, 1.28):
            head = build_cost_head({"name": "ridge", "z": z, "z_light": -0.5})
            head.fit(train)
            cost, _ = head.predict(texts)
            ratios.append(float(np.median(cost[:, 2] / cost[:, 0])))
        self.assertEqual(ratios, sorted(ratios), f"z를 올렸는데 비율이 안 커졌다: {ratios}")


class ResolutionTest(unittest.TestCase):
    def test_ridge_separates_episodes_within_a_family(self) -> None:
        """계열 평균은 계열 안에서 상수라 꼬리 컷이 작동하지 않는다.

        회귀 헤드는 같은 계열 안에서도 비용을 구분해야 한다.
        """

        from router.features import FAMILY_INDEX, family_codes

        train = train_set()
        flat = build_cost_head({"name": "family", "quantile": 0.75})
        sharp = build_cost_head({"name": "ridge", "z": 0.67})
        flat.fit(train)
        sharp.fit(train)
        texts = list(train.texts)
        codes = family_codes(texts)

        flat_cost, _ = flat.predict(texts)
        sharp_cost, _ = sharp.predict(texts)

        def spread(cost, mask) -> float:
            ratio = (cost[:, 2] / cost[:, 0])[mask]
            return float(ratio.max() / ratio.min())

        # K1 예산이 실제로 가는 고ROI 계열에서만 요구한다. 실측 배수는
        # sym_math 93.7 · mcq_en 4.30 · code_io 2.63 · mcq_ko 1.68이므로
        # 1.5배를 하한으로 둔다. K1을 아예 막는 계열(long_ctx 등)은 제외한다.
        for name in ("sym_math", "mcq_en", "code_io", "mcq_ko"):
            mask = codes == FAMILY_INDEX[name]
            if mask.sum() < 50:
                continue
            widened = spread(sharp_cost, mask) / spread(flat_cost, mask)
            self.assertGreater(
                widened, 1.5,
                f"{name}: 회귀 헤드가 계열 평균보다 {widened:.2f}배밖에 안 넓다",
            )


if __name__ == "__main__":
    unittest.main()
