from __future__ import annotations

from math import isclose
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_LVT_RATES = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
# A household counts as a winner (loser) only if its net income change is
# more than £1 above (below) zero; changes within ±£1 count as unchanged.
WINNER_THRESHOLD_GBP = 1.0
COUNTRY_ORDER = ("ENGLAND", "SCOTLAND", "WALES", "NORTHERN_IRELAND")
REGION_ORDER = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
    "NORTHERN_IRELAND",
)
FAMILY_TYPE_ORDER = (
    "Single, no children",
    "Couple, no children",
    "Lone parent",
    "Couple with children",
    "Single pensioner",
    "Pensioner couple",
)


def weighted_sum(values: pd.Series, weights: pd.Series) -> float:
    return float(np.sum(values.astype(float) * weights.astype(float)))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    total_weight = float(np.sum(weights.astype(float)))
    if total_weight == 0:
        return 0.0
    return weighted_sum(values, weights) / total_weight


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    # Lower weighted median: returns the first value whose cumulative weight
    # reaches half the total, with no interpolation between adjacent values.
    ordered = (
        pd.DataFrame({"value": values.astype(float), "weight": weights.astype(float)})
        .sort_values("value")
        .reset_index(drop=True)
    )
    if ordered.empty:
        return 0.0
    midpoint = ordered["weight"].sum() / 2
    return float(
        ordered.loc[ordered["weight"].cumsum() >= midpoint, "value"].iloc[0]
    )


def classify_family_type(adults: float, children: float, pensioners: float) -> str:
    if adults > 0 and pensioners >= adults and children == 0:
        return "Single pensioner" if adults <= 1 else "Pensioner couple"
    if adults <= 1:
        return "Lone parent" if children > 0 else "Single, no children"
    return "Couple with children" if children > 0 else "Couple, no children"


def make_rate_grid(
    required_rate: float,
    comparison_rates: Sequence[float] = DEFAULT_LVT_RATES,
) -> list[float]:
    unique_rates: list[float] = []
    for rate in sorted((required_rate, *comparison_rates)):
        if not any(isclose(rate, existing, abs_tol=1e-6) for existing in unique_rates):
            unique_rates.append(rate)
    return unique_rates


def format_rate_label(rate: float, required_rate: float) -> str:
    if isclose(rate, required_rate, abs_tol=1e-6):
        return f"{rate * 100:.2f}%"
    return f"{rate * 100:.1f}%"


def build_baseline_summary(df: pd.DataFrame) -> dict:
    weights = df["weight"]
    land_total = weighted_sum(df["land_value"], weights)
    household_land_total = weighted_sum(df["hh_land"], weights)
    corporate_land_total = weighted_sum(df["corp_land"], weights)
    property_wealth_total = weighted_sum(df["property_wealth"], weights)
    total_wealth_total = weighted_sum(df["total_wealth"], weights)
    return {
        "total_land_tn": round(land_total / 1e12, 2),
        "household_land_tn": round(household_land_total / 1e12, 2),
        "corporate_land_tn": round(corporate_land_total / 1e12, 2),
        "total_property_wealth_tn": round(property_wealth_total / 1e12, 2),
        "total_wealth_tn": round(total_wealth_total / 1e12, 2),
        "land_pct_of_property": round(household_land_total / property_wealth_total * 100, 1),
        "land_pct_of_wealth": round(land_total / total_wealth_total * 100, 1),
        "avg_land_per_household": round(weighted_mean(df["land_value"], weights)),
        "median_land_value": round(weighted_median(df["land_value"], weights)),
    }


def build_ons_comparison(
    baseline: dict,
    target_year: int,
    target_household_tn: float,
    target_corporate_tn: float,
    target_total_tn: float,
    reference_url: str | None = None,
) -> dict:
    return {
        "target_year": target_year,
        "target_label": str(target_year),
        "target_household_tn": round(target_household_tn, 2),
        "target_corporate_tn": round(target_corporate_tn, 2),
        "target_total_tn": round(target_total_tn, 2),
        "reference_url": reference_url,
        "model_2026_total_tn": baseline["total_land_tn"],
        "model_2026_household_tn": baseline["household_land_tn"],
        "model_2026_corporate_tn": baseline["corporate_land_tn"],
        "model_vs_target_pct": round(
            baseline["total_land_tn"] / target_total_tn * 100,
            1,
        ),
    }


def build_average_land_tables(df: pd.DataFrame) -> tuple[list[dict], list[dict], list[dict]]:
    avg_by_country = [
        {
            "group": "UK",
            "avg_land_value": round(weighted_mean(df["land_value"], df["weight"])),
        }
    ]
    for country in COUNTRY_ORDER:
        subset = df[df["country"] == country]
        if subset.empty:
            continue
        avg_by_country.append(
            {
                "group": country.replace("_", " ").title(),
                "avg_land_value": round(
                    weighted_mean(subset["land_value"], subset["weight"])
                ),
            }
        )

    avg_by_region = []
    for region in REGION_ORDER:
        subset = df[df["region"] == region]
        if subset.empty:
            continue
        label = region.replace("_", " ").title()
        if label == "East Of England":
            label = "East of England"
        if label == "Yorkshire":
            label = "Yorkshire and the Humber"
        avg_by_region.append(
            {
                "group": label,
                "avg_land_value": round(
                    weighted_mean(subset["land_value"], subset["weight"])
                ),
            }
        )

    avg_by_family_type = []
    for family_type in FAMILY_TYPE_ORDER:
        subset = df[df["family_type"] == family_type]
        if subset.empty:
            continue
        avg_by_family_type.append(
            {
                "group": family_type,
                "avg_land_value": round(
                    weighted_mean(subset["land_value"], subset["weight"])
                ),
            }
        )

    return avg_by_country, avg_by_region, avg_by_family_type


def build_distribution_by_decile(
    df: pd.DataFrame, decile_col: str = "income_decile"
) -> list[dict]:
    filtered = df[df[decile_col] > 0]
    total_land = weighted_sum(filtered["land_value"], filtered["weight"])
    rows = []
    for decile in sorted(filtered[decile_col].unique()):
        subset = filtered[filtered[decile_col] == decile]
        weights = subset["weight"]
        decile_land = weighted_sum(subset["land_value"], weights)
        rows.append(
            {
                "decile": int(decile),
                "avg_income": round(weighted_mean(subset["income"], weights)),
                "avg_land_value": round(weighted_mean(subset["land_value"], weights)),
                "avg_property_wealth": round(
                    weighted_mean(subset["property_wealth"], weights)
                ),
                "share_of_land_pct": round(decile_land / total_land * 100, 1),
            }
        )
    return rows


def build_impact_scenario_table(
    df: pd.DataFrame, decile_col: str = "income_decile"
) -> list[dict]:
    filtered = df[df[decile_col] > 0].copy()
    # Winners/losers use a ±£1 dead-band so the definition matches
    # analysis/deep_dive.py and analysis/extensions.py exactly.
    filtered["is_winner"] = (filtered["income_change"] > WINNER_THRESHOLD_GBP).astype(float)
    filtered["is_loser"] = (filtered["income_change"] < -WINNER_THRESHOLD_GBP).astype(float)
    filtered["is_unchanged"] = (
        filtered["income_change"].abs() <= WINNER_THRESHOLD_GBP
    ).astype(float)

    rows = []
    for decile in sorted(filtered[decile_col].unique()):
        subset = filtered[filtered[decile_col] == decile]
        weights = subset["weight"]
        avg_income_change = weighted_mean(subset["income_change"], weights)
        avg_baseline_income = weighted_mean(subset["baseline_income"], weights)
        rows.append(
            {
                "decile": int(decile),
                "avg_lvt": round(weighted_mean(subset["lvt"], weights)),
                "avg_council_tax_saved": round(
                    weighted_mean(subset["council_tax_saved"], weights)
                ),
                "avg_net_change": round(avg_income_change),
                "avg_income_change_pct": (
                    round(avg_income_change / avg_baseline_income * 100, 1)
                    if avg_baseline_income > 0
                    else None
                ),
                "avg_land_value": round(weighted_mean(subset["land_value"], weights)),
                "pct_winners": round(weighted_mean(subset["is_winner"], weights) * 100, 1),
                "pct_losers": round(weighted_mean(subset["is_loser"], weights) * 100, 1),
                "pct_unchanged": round(
                    weighted_mean(subset["is_unchanged"], weights) * 100,
                    1,
                ),
            }
        )
    return rows


def build_landless_summary(df: pd.DataFrame) -> dict:
    """Summarise households that own no land (``land_value <= 0``).

    These households pay no land value tax, so a council-tax-to-LVT swap
    leaves them with their council-tax saving. They sit at the bottom of the
    wealth distribution and are thinned out at the left edge of the
    wealth-decile chart, so we report them explicitly rather than dropping
    them by construction.
    """
    total_weight = float(df["weight"].sum())
    landless = df[df["land_value"] <= 0]
    landless_weight = float(landless["weight"].sum())
    if total_weight == 0 or landless_weight == 0:
        return {"share_pct": 0.0, "avg_net_change": 0, "pct_winners": 0.0}
    weights = landless["weight"]
    is_winner = (landless["income_change"] > WINNER_THRESHOLD_GBP).astype(float)
    return {
        "share_pct": round(landless_weight / total_weight * 100, 1),
        "avg_net_change": round(weighted_mean(landless["income_change"], weights)),
        "pct_winners": round(weighted_mean(is_winner, weights) * 100, 1),
    }


def build_council_tax_vs_lvt_table(
    df: pd.DataFrame, decile_col: str = "income_decile"
) -> list[dict]:
    filtered = df[df[decile_col] > 0]
    rows = []
    for decile in sorted(filtered[decile_col].unique()):
        subset = filtered[filtered[decile_col] == decile]
        weights = subset["weight"]
        avg_council_tax = weighted_mean(subset["council_tax"], weights)
        avg_lvt = weighted_mean(subset["lvt"], weights)
        difference = avg_lvt - avg_council_tax
        rows.append(
            {
                "decile": int(decile),
                "avg_council_tax": round(avg_council_tax),
                "avg_lvt": round(avg_lvt),
                "difference": round(difference),
                "change_pct": (
                    round(difference / avg_council_tax * 100, 1)
                    if avg_council_tax > 0
                    else None
                ),
            }
        )
    return rows


def build_revenue_by_rate(
    council_tax_revenue_bn: float,
    rate_rows: Iterable[dict],
) -> list[dict]:
    results = []
    for row in rate_rows:
        results.append(
            {
                "rate": row["rate"],
                "rate_pct": f"{row['rate']:.1%}",
                "lvt_revenue_bn": round(row["lvt_revenue_bn"], 1),
                "council_tax_bn": round(council_tax_revenue_bn, 1),
                "net_revenue_bn": round(row["lvt_revenue_bn"] - council_tax_revenue_bn, 1),
                "avg_per_household": round(row["avg_per_household"]),
            }
        )
    return results


def build_revenue_by_scope(scope_rows: Iterable[dict]) -> list[dict]:
    return [
        {
            "scope": row["scope"],
            "revenue_bn": round(row["revenue_bn"], 1),
            "avg_per_household": round(row["avg_per_household"]),
        }
        for row in scope_rows
    ]
