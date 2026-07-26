"""Paper figures for the LVT study.

Reads the committed ``results/lvt_results.json`` and writes each figure as a
PNG plus a sibling CSV of the plotted data into ``results/figures/``. No
simulation dependencies: this reproduces every chart without licensed data.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "lvt_results.json"
OUTDIR = ROOT / "results" / "figures"

NEUTRAL_RATE = "0.77%"
COUNCIL_TAX_BN = 57.6

fs.apply_style()
OUTDIR.mkdir(parents=True, exist_ok=True)


def _save(fig, df: pd.DataFrame, stem: str) -> None:
    fs.save(fig, OUTDIR / f"{stem}.png")
    df.to_csv(OUTDIR / f"{stem}.csv", index=False)
    print(f"wrote {stem}.png / {stem}.csv")


def _gain_colors(values) -> list[str]:
    return [fs.BLUE if v >= 0 else fs.GRAY for v in values]


def fig1_net_change_by_income_decile(data: dict) -> None:
    rows = data["impact_scenarios"][NEUTRAL_RATE]
    df = pd.DataFrame(rows)[["decile", "avg_net_change"]]
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.bar(df["decile"], df["avg_net_change"], color=_gain_colors(df["avg_net_change"]))
    ax.axhline(0, color=fs.BASELINE, linewidth=0.8)
    fs.decile_ax(ax, "Average net income change (£/year)")
    ax.set_title(
        f"Council tax to LVT swap at the {NEUTRAL_RATE} budget-neutral rate"
    )
    _save(fig, df, "fig1_net_change_by_income_decile")


def fig2_land_by_income_decile(data: dict) -> None:
    df = pd.DataFrame(data["distribution_by_decile"])[["decile", "avg_land_value"]]
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.bar(df["decile"], df["avg_land_value"], color=fs.BLUE)
    fs.decile_ax(ax, "Average household land value (£)")
    ax.set_title("Household land value by income decile")
    _save(fig, df, "fig2_land_by_income_decile")


def fig3_land_by_region(data: dict) -> None:
    df = pd.DataFrame(data["avg_land_by_region"]).sort_values(
        "avg_land_value", ascending=False
    )
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    fs.ranked_hbar(ax, list(df["group"]), list(df["avg_land_value"]))
    ax.set_xlabel("Average household land value")
    ax.set_title("Household land value by region")
    _save(fig, df, "fig3_land_by_region")


def fig4_land_by_family_type(data: dict) -> None:
    df = pd.DataFrame(data["avg_land_by_family_type"]).sort_values(
        "avg_land_value", ascending=False
    )
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    fs.ranked_hbar(ax, list(df["group"]), list(df["avg_land_value"]))
    ax.set_xlabel("Average household land value")
    ax.set_title("Household land value by family type")
    _save(fig, df, "fig4_land_by_family_type")


def fig5_net_change_by_wealth_decile(data: dict) -> None:
    rows = data["impact_scenarios_by_wealth"][NEUTRAL_RATE]
    df = pd.DataFrame(rows)[["decile", "avg_net_change"]]
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.bar(df["decile"], df["avg_net_change"], color=_gain_colors(df["avg_net_change"]))
    ax.axhline(0, color=fs.BASELINE, linewidth=0.8)
    fs.decile_ax(ax, "Average net income change (£/year)", xlabel="Wealth decile")
    ax.set_title(
        f"Council tax to LVT swap at the {NEUTRAL_RATE} rate, by wealth decile"
    )
    _save(fig, df, "fig5_net_change_by_wealth_decile")


def fig6_winners_losers_by_decile(data: dict) -> None:
    rows = data["impact_scenarios"][NEUTRAL_RATE]
    df = pd.DataFrame(rows)[["decile", "pct_winners", "pct_losers", "pct_unchanged"]]
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.bar(df["decile"], df["pct_winners"], color=fs.BLUE, label="Gain")
    ax.bar(
        df["decile"],
        df["pct_losers"],
        bottom=df["pct_winners"],
        color=fs.GRAY,
        label="Lose",
    )
    ax.bar(
        df["decile"],
        df["pct_unchanged"],
        bottom=df["pct_winners"] + df["pct_losers"],
        color=fs.LIGHT_GRAY,
        label="Unchanged",
    )
    fs.decile_ax(ax, "Share of households (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Winners and losers by income decile at the {NEUTRAL_RATE} rate")
    fs.legend_below(ax, 3)
    _save(fig, df, "fig6_winners_losers_by_decile")


def fig7_revenue_by_rate(data: dict) -> None:
    df = pd.DataFrame(data["revenue_by_rate"])[["rate", "lvt_revenue_bn"]]
    df["rate_pct"] = df["rate"] * 100
    fig, ax = plt.subplots(figsize=fs.SINGLE)
    ax.plot(df["rate_pct"], df["lvt_revenue_bn"], marker="o", color=fs.BLUE, label="LVT revenue")
    ax.axhline(
        COUNCIL_TAX_BN,
        color=fs.GRAY,
        linestyle="--",
        linewidth=1.0,
        label=f"Council tax revenue (£{COUNCIL_TAX_BN}bn)",
    )
    ax.set_xlabel("LVT rate (% of land value)")
    ax.set_ylabel("Revenue (£bn/year)")
    ax.set_title("LVT revenue by rate")
    ax.legend()
    _save(fig, df[["rate", "rate_pct", "lvt_revenue_bn"]], "fig7_revenue_by_rate")


def fig8_poverty_gini_by_rate(data: dict) -> None:
    scen = data["poverty_gini"]["scenarios"]
    rows = [
        {
            "rate_label": label,
            "rate_pct": float(label.rstrip("%")),
            "poverty_bhc_change": v["poverty_bhc_change"],
            "poverty_ahc_change": v["poverty_ahc_change"],
            "gini_change": v["gini_change"],
        }
        for label, v in scen.items()
    ]
    df = pd.DataFrame(rows).sort_values("rate_pct")
    neutral = 0.77
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fs.TWOPANEL)

    ax1.plot(df["rate_pct"], df["poverty_bhc_change"], marker="o", color=fs.BLUE, label="Poverty (BHC)")
    ax1.plot(df["rate_pct"], df["poverty_ahc_change"], marker="s", color=fs.TEAL, label="Poverty (AHC)")
    ax1.axhline(0, color=fs.BASELINE, linewidth=0.8)
    ax1.set_xlabel("LVT rate (% of land value)")
    ax1.set_ylabel("Change in poverty rate (pp)")
    ax1.set_title("Poverty")
    ax1.legend()

    ax2.plot(df["rate_pct"], df["gini_change"], marker="o", color=fs.BLUE)
    ax2.axhline(0, color=fs.BASELINE, linewidth=0.8)
    ax2.set_xlabel("LVT rate (% of land value)")
    ax2.set_ylabel("Change in Gini coefficient")
    ax2.set_title("Income inequality")

    for ax in (ax1, ax2):
        ax.axvline(neutral, color=fs.GRAY, linestyle=":", linewidth=1.0)
        ax.annotate(
            "budget-neutral\n(0.77%)",
            xy=(neutral, ax.get_ylim()[1]),
            xytext=(neutral + 0.12, ax.get_ylim()[1] * 0.86),
            fontsize=8,
            color=fs.INK2,
        )
    _save(fig, df, "fig8_poverty_gini_by_rate")


def main() -> None:
    data = json.loads(RESULTS.read_text())
    fig1_net_change_by_income_decile(data)
    fig2_land_by_income_decile(data)
    fig3_land_by_region(data)
    fig4_land_by_family_type(data)
    fig5_net_change_by_wealth_decile(data)
    fig6_winners_losers_by_decile(data)
    fig7_revenue_by_rate(data)
    fig8_poverty_gini_by_rate(data)


if __name__ == "__main__":
    main()
