# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""공식 hashed 표현 + 참가자 예산 할당기 결합의 계약 테스트."""

from __future__ import annotations

import json
import unittest

import numpy as np

from baselines.hash_regex import raw_feature_vector
from router.data import load_dataset
from router.hash_features import extract_hash_features
from router.heads import build_cost_head, build_score_head

_DATA = {}


def datasets():
    if not _DATA:
        _DATA["train"] = load_dataset("train")
        _DATA["dev"] = load_dataset("dev")
    return _DATA["train"], _DATA["dev"]


class HashFeatureTest(unittest.TestCase):
    def test_text_contract_exactly_matches_public_baseline_features(self) -> None:
        """단일 prompt/message에서 참가자 특징은 공개 기준과 같아야 한다."""

        train, _dev = datasets()
        episodes = train.inputs.episodes[:32]
        expected = np.asarray(
            [raw_feature_vector(episode, 256) for episode in episodes], dtype=float
        )
        actual = extract_hash_features(train.texts[:32], 256)
        np.testing.assert_array_equal(actual, expected)

    def test_is_deterministic_and_rejects_invalid_bins(self) -> None:
        texts = ("Prove x^2 >= 0.", "파이썬 코드의 복잡도를 분석해줘")
        np.testing.assert_array_equal(
            extract_hash_features(texts, 64), extract_hash_features(texts, 64)
        )
        with self.assertRaises(ValueError):
            extract_hash_features(texts, 63)


class HashHeadTest(unittest.TestCase):
    def test_score_and_cost_round_trip(self) -> None:
        train, dev = datasets()
        score = build_score_head("hash_ridge")
        cost = build_cost_head("hash_ridge")
        score.fit(train)
        cost.fit(train)
        texts = dev.texts[:80]
        expected_score = score.predict(texts)
        expected_cost, expected_sd = cost.predict(texts)

        revived_score = build_score_head("hash_ridge")
        revived_cost = build_cost_head("hash_ridge")
        revived_score.load_state(json.loads(json.dumps(score.state())))
        revived_cost.load_state(json.loads(json.dumps(cost.state())), train.policy)
        np.testing.assert_allclose(expected_score, revived_score.predict(texts), atol=1e-12)
        actual_cost, actual_sd = revived_cost.predict(texts)
        np.testing.assert_allclose(expected_cost, actual_cost, atol=1e-12)
        np.testing.assert_allclose(expected_sd, actual_sd, atol=1e-12)

    def test_ranges_monotonic_costs_and_adversarial_finiteness(self) -> None:
        train, _dev = datasets()
        score = build_score_head("hash_ridge")
        cost = build_cost_head("hash_ridge")
        score.fit(train)
        cost.fit(train)
        texts = ("", "9" * 5_000, "\\frac" * 2_000, "가" * 20_000)
        predicted = score.predict(texts)
        predicted_cost, sd = cost.predict(texts)
        self.assertTrue(((predicted >= 0) & (predicted <= 1)).all())
        self.assertTrue(np.isfinite(predicted_cost).all())
        self.assertTrue(np.isfinite(sd).all())
        self.assertTrue((predicted_cost > 0).all())
        self.assertTrue((predicted_cost[:, 1] >= predicted_cost[:, 0]).all())
        self.assertTrue((predicted_cost[:, 2] >= predicted_cost[:, 1]).all())


if __name__ == "__main__":
    unittest.main()
