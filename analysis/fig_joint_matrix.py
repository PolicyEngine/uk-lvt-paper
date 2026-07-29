"""Heatmap of mean net income change by income decile x wealth decile.

Reads results/submission_extras.json (written by submission_extras.py) and
writes results/figures/fig_joint_income_wealth.png/.csv in the shared house
style. Cells holding under 0.1 per cent of households are greyed out.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "figures"

fs.apply_style()


def main():
    data = json.loads((ROOT / "results" / "submission_extras.json").read_text())
    block = data["joint_income_wealth"]
    change = np.array(
        [[np.nan if v is None else v for v in row] for row in block["mean_net_change"]],
        dtype=float,
    )
    share = np.array(block["population_share_pct"], dtype=float)
    change_masked = np.where(share < 0.1, np.nan, change)

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    vmax = np.nanmax(np.abs(change_masked))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(change_masked, cmap=fs.DIVERGING, norm=norm, aspect="auto",
                   origin="lower")
    ax.set_xticks(range(10), [str(i) for i in range(1, 11)])
    ax.set_yticks(range(10), [str(i) for i in range(1, 11)])
    ax.set_xlabel("Wealth decile")
    ax.set_ylabel("Income decile")
    for i in range(10):
        for j in range(10):
            v = change_masked[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+,.0f}".replace(",", " "),
                    ha="center", va="center", fontsize=6.5, color=fs.INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean net income change (£/year)")
    ax.set_title("Mean net income change by income and wealth decile,\n"
                 "budget-neutral reference scenario")

    df = pd.DataFrame(change, index=[f"inc{i}" for i in range(1, 11)],
                      columns=[f"wlth{j}" for j in range(1, 11)])
    fs.save(fig, OUTDIR / "fig_joint_income_wealth.png")
    df.to_csv(OUTDIR / "fig_joint_income_wealth.csv")
    print("wrote fig_joint_income_wealth.png/.csv")
    print("rank correlation:", block["rank_correlation_income_wealth"])


if __name__ == "__main__":
    main()
