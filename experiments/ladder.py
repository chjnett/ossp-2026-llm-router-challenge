# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""점수가 어디서 나오는지 레버별로 분해한다 (plan/GOAL.md 근거).

Train에서 배우고 Dev에서 평가한다. 실제 관측값을 예측 대신 넣은 줄은
'상한'으로 따로 표시한다 — 도달 가능한 값이 아니라 그 레버의 크기다.

    PYTHONPATH=src python3 experiments/ladder.py
"""

from __future__ import annotations

import numpy as np

from router.allocate import allocate
from router.data import MODEL_IDS, TIERS, load_dataset
from router.features import FAMILIES, family_codes
from router.harness import evaluate
from router.pipeline import Config, predict

MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}


def run(name: str, dev, s_hat, c_hat, *, util, allow=None, ceiling=False) -> None:
    picks = {
        t: allocate(
            s_hat, c_hat, multiplier=MULT[t], util=util, allow=allow, keys=dev.keys
        ).picks
        for t in TIERS
    }
    ev = evaluate(dev, picks)
    cells = " | ".join(
        f"{float(ev.tiers[t].score):.4f}/{float(ev.tiers[t].budget_ratio):5.3f}"
        f"{'✗' if not ev.tiers[t].passed else ' '}"
        for t in TIERS
    )
    mark = "  (상한)" if ceiling else ""
    print(f"  {name:38s} {cells}   가중 {float(ev.final_score):.4f}{mark}")


def main() -> None:
    train, dev = load_dataset("train"), load_dataset("dev")
    fam_dev = family_codes(dev.texts)

    # 예측은 등록된 헤드를 그대로 쓴다. 여기서만 쓰는 별도 구현을 두면
    # 러너와 숫자가 갈려 어느 쪽이 맞는지 알 수 없게 된다 (RULES D6).
    prediction = predict(Config(id="ladder-a1"), train, dev.texts)
    s_fam, c_fam = prediction.s_hat, prediction.c_hat

    # 계열별 K1 ROI를 Train에서 계산해 하위 계열의 K1을 막는다
    roi = np.zeros(len(FAMILIES))
    fam_train = family_codes(train.texts)
    for f in range(len(FAMILIES)):
        m = fam_train == f
        if not m.any():
            continue
        gain = train.score[m, 2].mean() - train.score[m, 0].mean()
        extra = train.cost[m, 2].mean() - train.cost[m, 0].mean()
        roi[f] = gain / extra if extra > 0 else 0.0
    blocked = {FAMILIES[f] for f in range(len(FAMILIES)) if roi[f] < 1.0}
    allow_k1 = np.ones((len(dev), 3), dtype=bool)
    allow_k1[:, 2] = roi[fam_dev] >= 1.0

    print(f"\nTrain 계열별 K1 ROI (크레딧당 점수 상승):")
    for f in np.argsort(-roi):
        n = int((fam_train == f).sum())
        flag = "" if roi[f] >= 1.0 else "   ← K1 차단"
        print(f"    {FAMILIES[f]:10s} n={n:5d}  ROI={roi[f]:6.2f}{flag}")

    print(f"\n{'':40s} {'fast':>13s} | {'balanced':>13s} | {'premium':>13s}")
    print("\n[실현 가능 경로]  Train에서 배우고 Dev에서 평가, util=0.90")
    run("A0  all-light", dev, np.zeros_like(dev.score), dev.cost, util=0.90,
        allow=np.array([[True, False, False]] * len(dev)))
    run("A1  계열 평균 점수+비용", dev, s_fam, c_fam, util=0.90)
    run("A2  A1 + 저ROI 계열 K1 차단", dev, s_fam, c_fam, util=0.90, allow=allow_k1)

    print("\n[레버 크기]  실제 관측값을 넣어 각 레버의 최대치를 잰다, util=0.90")
    run("B1  점수만 완벽 (비용은 계열평균)", dev, dev.score, c_fam, util=0.90, ceiling=True)
    run("B2  비용만 완벽 (점수는 계열평균)", dev, s_fam, dev.cost, util=0.90, ceiling=True)
    run("B3  둘 다 완벽 = oracle", dev, dev.score, dev.cost, util=0.90, ceiling=True)

    print("\n[안전 마진의 값어치]  oracle을 사용률만 바꿔가며")
    for util in (1.00, 0.95, 0.90, 0.85, 0.80):
        run(f"    util={util:.2f}", dev, dev.score, dev.cost, util=util, ceiling=True)

    print("\n[상위 3개 계열만 맞히면]  나머지는 전부 light 고정, util=0.90")
    top3 = set(np.argsort(-roi)[:3].tolist())
    allow_top3 = np.zeros((len(dev), 3), dtype=bool)
    allow_top3[:, 0] = True
    for f in top3:
        allow_top3[fam_dev == f, 1:] = True
    print(f"    대상 계열: {[FAMILIES[f] for f in top3]}  "
          f"({int(sum(fam_dev == f for f in top3).sum())}/{len(dev)}문항)")
    run("C1  상위3계열 + 계열평균 예측", dev, s_fam, c_fam, util=0.90, allow=allow_top3)
    run("C2  상위3계열 + 완벽 예측", dev, dev.score, dev.cost, util=0.90,
        allow=allow_top3, ceiling=True)


if __name__ == "__main__":
    main()
