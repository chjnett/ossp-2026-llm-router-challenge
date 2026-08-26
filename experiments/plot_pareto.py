# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0
"""Pareto figure for the report/video: Train-CV quality vs gate stress risk.

T43 strictly improves T38 Train-CV at the same measured gate risk and is the
highest-CV configuration that passes the complete C4 gate.
    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src python3 experiments/plot_pareto.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# prior: Apple SD Gothic Neo lacks U+2212; use ASCII minus in labels.
plt.rcParams["axes.unicode_minus"] = False


# label -> (Train-CV, gate_overs, passed_C4, marker_label)
POINTS = [
    # (cv, overs, passed, tag)
    (0.6663, 26, True, "T38"),
    (0.6667, 26, True, "T43 selected"),
    (0.6672, 31, False, "T40 fast h0.85"),
    (0.6670, 70, False, "T41 prem q0.65"),
    (0.6678, 75, False, "T42 both"),
    (0.6545, 3, False, "T39 score-blend"),
]


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for cv, overs, passed, tag in POINTS:
        if tag == "T43 selected":
            color, marker, size = "#087f5b", "D", 105
        elif passed:
            color, marker, size = "#868e96", "o", 75
        else:
            color, marker, size = "#c62828", "o", 75
        ax.scatter(cv, overs, s=size, color=color, marker=marker, zorder=3)
        dy = -11 if tag == "T43 selected" else 9
        ax.annotate(tag, (cv, overs), textcoords="offset points",
                    xytext=(6, dy), fontsize=9)
    # C4 boundary
    ax.axhline(30, color="#455a64", linestyle="--", linewidth=1.2)
    ax.text(0.6542, 31, "C4 gate: overs <= 30 / 12,000", fontsize=8,
            color="#455a64", va="bottom")
    # frontier hint
    ax.axvspan(0.6662, 0.6680, color="#2e7d32", alpha=0.06)
    ax.set_xlabel("Train-CV weighted final (higher = better)")
    ax.set_ylabel("Gate overs / 12,000 (lower = safer)")
    ax.annotate(
        "same risk, higher CV",
        xy=(0.6667, 26),
        xytext=(0.6608, 48),
        arrowprops={"arrowstyle": "->", "color": "#087f5b", "lw": 1.2},
        fontsize=9,
        color="#087f5b",
        weight="bold",
    )
    ax.set_title("Pareto: T43 is the highest-CV configuration passing C4")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "plan/submission/figures/pareto-frontier.png"
    fig.savefig(out, dpi=160)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
