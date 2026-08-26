# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

"""촬영 화면에 맞는 짧은 self-check 표를 출력한다."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print()
    print("tier       score    budget    status")
    for tier in ("fast", "balanced", "premium"):
        row = report["tiers"][tier]
        ratio = float(row["total_cost"]) / float(row["budget_limit"])
        print(f"{tier:<10} {float(row['tier_score']):.4f}   {ratio:6.1%}    PASS")
    print(f"weighted   {float(report['final_score']):.4f}")


if __name__ == "__main__":
    main()
