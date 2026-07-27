"""Extended analysis for the LVT paper: incidence detail, progressivity,
uncertainty, and alternative reform packages.

Writes ``results/deep_dive.json``. Everything derives from a single baseline
simulation plus the closed-form swap arithmetic validated in
``uk_lvt/pipeline_direct.py`` (net income change is exactly
``council tax saved - rate x land``, with no benefit interactions).

Sections
--------
``etr_by_property_value``
    Effective tax rate of council tax and of the LVT as a share of property
    value, by property value band. This is the regressivity that motivates the
    reform, in the form used by Adam et al. (2020).

``progressivity``
    Kakwani and Suits indices for council tax and the LVT, computed against
    both the income and the wealth ranking. Quantifies the paper's central
    claim that the two rankings tell different stories.

``by_tenure``
    Incidence split by tenure. Relevant to the pass-through question raised by
    Nielsson et al. (2024): renters gain their whole council tax bill under the
    fixed-rent assumption, and this section sizes that channel.

``decile_uncertainty``
    Bootstrap confidence intervals on the average net income change by income
    decile, resampling households with replacement.

``losers``
    Characteristics of losing households in the bottom three income deciles,
    against gainers in the same deciles.

``recycling``
    LVT with the net revenue returned as an equal per-person dividend --- the
    reform package actually proposed by George (1879), as against the
    unrecycled levy in the main rate grid.

``hvcts``
    The government's High Value Council Tax Surcharge (April 2028) simulated on
    the same microdata, as a comparator instrument.

``alternative_targets``
    Replacing council tax plus stamp duty land tax; and folding Northern
    Ireland domestic rates into the replacement.

``exempt_band``
    Revenue-neutral rates and incidence with a per-household exempt band of
    land value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = "hf://policyengine/policyengine-uk-data-private/enhanced_frs_2023_24.h5@1.56.14"
YEAR = 2026
OUTPUT = ROOT / "results" / "deep_dive.json"

BOOTSTRAP_DRAWS = 500
BOOTSTRAP_SEED = 20260724

PROPERTY_BANDS = [
    (1, 150_000, "Under 150k"),
    (150_000, 200_000, "150-200k"),
    (200_000, 300_000, "200-300k"),
    (300_000, 400_000, "300-400k"),
    (400_000, 500_000, "400-500k"),
    (500_000, 750_000, "500-750k"),
    (750_000, 1_000_000, "750k-1m"),
    (1_000_000, 2_000_000, "1-2m"),
    (2_000_000, np.inf, "Over 2m"),
]

# High Value Council Tax Surcharge, Autumn Budget 2025: England only, from
# April 2028, on 2026 values. Flat annual amounts by band.
HVCTS_BANDS = [
    (2_000_000, 2_500_000, 2_500),
    (2_500_000, 3_500_000, 3_500),
    (3_500_000, 5_000_000, 5_000),
    (5_000_000, np.inf, 7_500),
]

EXEMPT_BANDS = (50_000, 100_000, 200_000)
# Rates above the budget-neutral rate for the dividend-recycling scenarios;
# the solved budget-neutral rate itself is prepended at runtime (it is not
# hard-coded so that it always matches the central scenario exactly).
DIVIDEND_RATES = (0.01, 0.015, 0.02, 0.03, 0.05)


def _v(sim, name, year=YEAR, dtype=np.float64):
    return np.asarray(sim.calculate(name, year).values, dtype=dtype)


def _wmean(x, w):
    tw = np.sum(w)
    return float(np.sum(x * w) / tw) if tw > 0 else float("nan")


def _frac_rank(y, w):
    """Weighted fractional rank of y (midpoint convention)."""
    order = np.argsort(y, kind="stable")
    w_sorted = w[order]
    cum = np.cumsum(w_sorted) - 0.5 * w_sorted
    rank = np.empty_like(cum)
    rank[order] = cum / np.sum(w)
    return rank


def concentration_index(x, rank, w):
    """Concentration index of x against a given fractional ranking."""
    mu = _wmean(x, w)
    if mu == 0:
        return float("nan")
    return float(2 * np.sum(w * (x - mu) * (rank - _wmean(rank, w))) / (np.sum(w) * mu))


def gini(x, w):
    return concentration_index(x, _frac_rank(x, w), w)


def suits_index(tax, y, w):
    """Suits progressivity index of `tax` against the distribution of `y`."""
    order = np.argsort(y, kind="stable")
    t = (tax * w)[order]
    inc = (y * w)[order]
    if t.sum() <= 0 or inc.sum() <= 0:
        return float("nan")
    x = np.concatenate([[0.0], np.cumsum(inc) / inc.sum()])
    z = np.concatenate([[0.0], np.cumsum(t) / t.sum()])
    area = np.trapz(z, x)
    return float(1 - 2 * area)


def poverty_rates(df, delta):
    """Poverty rates (% of individuals, HBAI convention), baseline-fixed lines."""
    pw = df["person_weight"].values
    bhc = df["equiv_bhc"].values + delta / df["equivalisation_bhc"].values
    ahc = df["equiv_ahc"].values + delta / df["equivalisation_ahc"].values
    line_bhc = df["poverty_line_bhc"].values / df["equivalisation_bhc"].values
    line_ahc = df["poverty_line_ahc"].values / df["equivalisation_ahc"].values
    return (
        100 * float(np.sum((bhc < line_bhc) * pw) / np.sum(pw)),
        100 * float(np.sum((ahc < line_ahc) * pw) / np.sum(pw)),
    )


def income_gini(df, delta=0.0):
    """Person-weighted Gini of equivalised HBAI household net income (BHC)."""
    equiv = df["equiv_bhc"].values + delta / df["equivalisation_bhc"].values
    return gini(equiv, df["person_weight"].values)


def equal_weight_deciles(values, weights, n=10):
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order]) / np.sum(weights)
    d = np.searchsorted(np.arange(1, n) / n, cw, side="left") + 1
    out = np.empty(len(values), dtype=int)
    out[order] = np.minimum(d, n)
    return out


def load():
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=DATASET)
    df = pd.DataFrame(
        {
            "weight": _v(sim, "household_weight"),
            "income": _v(sim, "household_net_income"),
            "council_tax": _v(sim, "council_tax"),
            "council_tax_benefit": np.asarray(
                sim.calculate(
                    "council_tax_benefit", YEAR, map_to="household"
                ).values,
                dtype=np.float64,
            ),
            "domestic_rates": _v(sim, "domestic_rates"),
            "sdlt": _v(sim, "stamp_duty_land_tax"),
            "land": _v(sim, "land_value"),
            "hh_land": _v(sim, "household_land_value"),
            "property_wealth": _v(sim, "property_wealth"),
            "residential_value": _v(sim, "residential_property_value"),
            "main_residence_value": _v(sim, "main_residence_value"),
            "total_wealth": _v(sim, "total_wealth"),
            "income_decile": _v(sim, "household_income_decile", dtype=int),
            "in_poverty_bhc": _v(sim, "in_poverty_bhc"),
            "in_poverty_ahc": _v(sim, "in_poverty_ahc"),
            "equiv_bhc": _v(sim, "equiv_hbai_household_net_income"),
            "equiv_ahc": _v(sim, "equiv_hbai_household_net_income_ahc"),
            "equivalisation_bhc": _v(sim, "household_equivalisation_bhc"),
            "equivalisation_ahc": _v(sim, "household_equivalisation_ahc"),
            "poverty_line_bhc": _v(sim, "poverty_line_bhc"),
            "poverty_line_ahc": _v(sim, "poverty_line_ahc"),
        }
    )
    df["council_tax_saved"] = (
        df["council_tax"] - df["council_tax_benefit"]
    )
    df["country"] = np.asarray(sim.calculate("country", YEAR).values).astype(str)
    df["region"] = np.asarray(sim.calculate("region", YEAR).values).astype(str)
    df["tenure"] = np.asarray(sim.calculate("tenure_type", YEAR).values).astype(str)
    df["wealth_decile"] = equal_weight_deciles(
        df["total_wealth"].values, df["weight"].values
    )

    # person-level aggregates
    age = _v(sim, "age")
    sp = _v(sim, "is_SP_age")
    hh_of_person = np.asarray(sim.calculate("household_id", YEAR, map_to="person").values)
    ids = np.asarray(sim.calculate("household_id", YEAR).values)
    g = pd.DataFrame(
        {"hh": hh_of_person, "age": age, "sp": sp, "one": 1.0}
    ).groupby("hh")
    df["people"] = g["one"].sum().reindex(ids).fillna(0).values
    df["max_age"] = g["age"].max().reindex(ids).fillna(0).values
    df["any_pensioner"] = (g["sp"].max().reindex(ids).fillna(0).values > 0)
    # Person weight: poverty is a share of individuals (HBAI convention) and
    # the income Gini is person-weighted over equivalised income.
    df["person_weight"] = df["weight"] * df["people"]
    return sim, df


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def etr_by_property_value(df, rate, value_col="residential_value"):
    """Council tax and LVT as a share of property value, property owners only.

    Households owning no residential property are excluded: their council tax
    bill divided by a property value of zero is not an effective rate. They are
    reported as a separate memo row. ``value_col`` selects the base:
    ``residential_value`` (all residential property, including second and let
    homes) or ``main_residence_value`` (main residence only).
    """
    w = df["weight"].values
    val = df[value_col].values
    ct = df["council_tax_saved"].values
    lvt = rate * df["land"].values
    rows = []
    nonowner = val <= 0
    if nonowner.any():
        rows.append(
            {
                "band": "No residential property (memo)",
                "share_of_households_pct": round(100 * np.sum(w[nonowner]) / np.sum(w), 1),
                "avg_property_value": 0,
                "avg_council_tax": round(_wmean(ct[nonowner], w[nonowner])),
                "avg_lvt": round(_wmean(lvt[nonowner], w[nonowner])),
                "council_tax_pct_of_value": None,
                "lvt_pct_of_value": None,
            }
        )
    for lo, hi, label in PROPERTY_BANDS:
        m = (val >= lo) & (val < hi)
        if not m.any() or np.sum(w[m]) == 0:
            continue
        mean_val = _wmean(val[m], w[m])
        rows.append(
            {
                "band": label,
                "share_of_households_pct": round(100 * np.sum(w[m]) / np.sum(w), 1),
                "avg_property_value": round(mean_val),
                "avg_council_tax": round(_wmean(ct[m], w[m])),
                "avg_lvt": round(_wmean(lvt[m], w[m])),
                "council_tax_pct_of_value": round(100 * _wmean(ct[m], w[m]) / mean_val, 3),
                "lvt_pct_of_value": round(100 * _wmean(lvt[m], w[m]) / mean_val, 3),
            }
        )
    return rows


def top_tail_reconciliation(df):
    """Households above £2m under the two property-value definitions.

    The ETR table uses ``residential_value`` (all residential property,
    including second and let homes), while the HVCTS comparator and the OBR
    costing count *main residences* over £2m (England only). This memo makes
    the gap explicit so the two figures in the paper reconcile.
    """
    w = df["weight"].values
    england = df["country"].values == "ENGLAND"

    def count_k(mask):
        return round(float(np.sum(w[mask])) / 1e3)

    return {
        "households_residential_over_2m_thousands": count_k(
            df["residential_value"].values >= 2_000_000
        ),
        "households_main_residence_over_2m_thousands": count_k(
            df["main_residence_value"].values >= 2_000_000
        ),
        "households_main_residence_over_2m_england_thousands": count_k(
            (df["main_residence_value"].values >= 2_000_000) & england
        ),
        "obr_hvcts_properties_thousands": 165,
        "note": (
            "residential_value includes second and let homes aggregated to "
            "the owning household, so it exceeds counts of main residences; "
            "the OBR HVCTS figure is English main residences only."
        ),
    }


def progressivity(df, rate):
    w = df["weight"].values
    pw = df["person_weight"].values
    ct = df["council_tax_saved"].values
    lvt = rate * df["land"].values
    # Income side: equivalised HBAI net income, person-weighted, matching the
    # Gini convention used elsewhere in the paper. Wealth side: total
    # household wealth, household-weighted (ONS WAS convention).
    inc = df["equiv_bhc"].values
    wealth = df["total_wealth"].values
    rank_inc = _frac_rank(inc, pw)
    rank_wealth = _frac_rank(wealth, w)
    g_inc = gini(inc, pw)
    g_wealth = gini(wealth, w)
    # Sensitivity: wealth indices are not well defined over negative values,
    # so recompute with wealth bottom-coded at zero.
    wealth_bc = np.maximum(wealth, 0.0)
    rank_wealth_bc = _frac_rank(wealth_bc, w)
    g_wealth_bc = gini(wealth_bc, w)

    def block(tax, label):
        return {
            "tax": label,
            "concentration_index_income_rank": round(
                concentration_index(tax, rank_inc, pw), 4
            ),
            "kakwani_income": round(concentration_index(tax, rank_inc, pw) - g_inc, 4),
            "suits_income": round(suits_index(tax, inc, pw), 4),
            "concentration_index_wealth_rank": round(
                concentration_index(tax, rank_wealth, w), 4
            ),
            "kakwani_wealth": round(
                concentration_index(tax, rank_wealth, w) - g_wealth, 4
            ),
            "suits_wealth": round(suits_index(tax, wealth, w), 4),
            "kakwani_wealth_bottom_coded": round(
                concentration_index(tax, rank_wealth_bc, w) - g_wealth_bc, 4
            ),
            "suits_wealth_bottom_coded": round(suits_index(tax, wealth_bc, w), 4),
        }

    return {
        "gini_income": round(g_inc, 4),
        "gini_wealth": round(g_wealth, 4),
        "gini_wealth_bottom_coded": round(g_wealth_bc, 4),
        "note": (
            "Kakwani = concentration index of the tax minus the Gini of the "
            "ranking variable; positive is progressive. Suits is computed "
            "against cumulative shares of the ranking variable. Income side "
            "uses equivalised HBAI net income, person-weighted; wealth side "
            "uses total household wealth, household-weighted. Wealth indices "
            "are also reported with wealth bottom-coded at zero, since "
            "cumulative-share indices are not well defined over negative "
            "values."
        ),
        "taxes": [block(ct, "Council tax (net of CTR)"), block(lvt, "LVT at budget-neutral rate")],
    }


def by_tenure(df, rate):
    w = df["weight"].values
    ct = df["council_tax_saved"].values
    lvt = rate * df["land"].values
    delta = ct - lvt
    land = df["land"].values
    rows = []
    for tenure in sorted(set(df["tenure"])):
        m = (df["tenure"] == tenure).values
        rows.append(
            {
                "tenure": tenure,
                "share_of_households_pct": round(100 * np.sum(w[m]) / np.sum(w), 1),
                "avg_land_value": round(_wmean(land[m], w[m])),
                "avg_council_tax_saved": round(_wmean(ct[m], w[m])),
                "avg_lvt": round(_wmean(lvt[m], w[m])),
                "avg_net_change": round(_wmean(delta[m], w[m])),
                "pct_winners": round(100 * np.sum((delta[m] > 1) * w[m]) / np.sum(w[m]), 1),
                "share_of_lvt_paid_pct": round(
                    100 * np.sum(lvt[m] * w[m]) / np.sum(lvt * w), 1
                ),
                "share_of_land_pct": round(100 * np.sum(land[m] * w[m]) / np.sum(land * w), 1),
                "aggregate_net_change_bn": round(float(np.sum(delta[m] * w[m])) / 1e9, 2),
            }
        )
    return rows


def decile_uncertainty(df, rate, draws=BOOTSTRAP_DRAWS):
    """Bootstrap CIs on average net change by income decile."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    w = df["weight"].values
    delta = df["council_tax_saved"].values - rate * df["land"].values
    dec = df["income_decile"].values
    n = len(df)
    keep = dec > 0
    draws_by_decile = {d: [] for d in range(1, 11)}
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        dw = w[idx]
        dd = dec[idx]
        dv = delta[idx]
        for d in range(1, 11):
            m = dd == d
            if m.any() and np.sum(dw[m]) > 0:
                draws_by_decile[d].append(_wmean(dv[m], dw[m]))
    rows = []
    for d in range(1, 11):
        m = keep & (dec == d)
        point = _wmean(delta[m], w[m])
        arr = np.array(draws_by_decile[d])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append(
            {
                "decile": d,
                "avg_net_change": round(point),
                "ci_low": round(float(lo)),
                "ci_high": round(float(hi)),
                "se": round(float(arr.std(ddof=1))),
                "significant_at_5pct": bool(lo > 0 or hi < 0),
            }
        )
    return {"draws": draws, "seed": BOOTSTRAP_SEED, "by_decile": rows}


def losers(df, rate):
    w = df["weight"].values
    delta = df["council_tax_saved"].values - rate * df["land"].values
    bottom = (df["income_decile"].values >= 1) & (df["income_decile"].values <= 3)
    lose = bottom & (delta < -1)
    gain = bottom & (delta > 1)

    def profile(mask, label):
        ww = w[mask]
        if np.sum(ww) == 0:
            return None
        tenure = (
            pd.Series(df["tenure"].values[mask])
            .groupby(pd.Series(df["tenure"].values[mask]))
            .size()
        )
        tenure_share = {
            t: round(100 * float(np.sum(ww[df["tenure"].values[mask] == t]) / np.sum(ww)), 1)
            for t in sorted(set(df["tenure"].values[mask]))
        }
        region = df["region"].values[mask]
        region_share = {
            r: round(100 * float(np.sum(ww[region == r]) / np.sum(ww)), 1)
            for r in sorted(set(region))
        }
        top_regions = dict(
            sorted(region_share.items(), key=lambda kv: -kv[1])[:4]
        )
        return {
            "group": label,
            "share_of_bottom_three_deciles_pct": round(
                100 * float(np.sum(ww) / np.sum(w[bottom])), 1
            ),
            "avg_net_change": round(_wmean(delta[mask], ww)),
            "avg_land_value": round(_wmean(df["land"].values[mask], ww)),
            "avg_property_value": round(_wmean(df["residential_value"].values[mask], ww)),
            "pct_with_pensioner": round(
                100 * float(np.sum(ww[df["any_pensioner"].values[mask]]) / np.sum(ww)), 1
            ),
            "avg_oldest_member_age": round(_wmean(df["max_age"].values[mask], ww), 1),
            "tenure_shares_pct": tenure_share,
            "top_regions_pct": top_regions,
        }

    return {
        "definition": "Income deciles 1-3 at the budget-neutral rate",
        "profiles": [profile(lose, "Losers"), profile(gain, "Gainers")],
    }


def recycling(df, ct_revenue_bn, central_rate):
    """LVT with net revenue returned as an equal per-person dividend."""
    w = df["weight"].values
    ct = df["council_tax_saved"].values
    land = df["land"].values
    people = df["people"].values
    total_people = float(np.sum(people * w))
    pw = df["person_weight"].values
    base_bhc = 100 * float(np.sum(df["in_poverty_bhc"].values * pw) / np.sum(pw))
    base_ahc = 100 * float(np.sum(df["in_poverty_ahc"].values * pw) / np.sum(pw))
    base_gini = income_gini(df)
    dec = df["income_decile"].values

    rows = []
    for rate in (central_rate,) + DIVIDEND_RATES:
        lvt = rate * land
        net_bn = (float(np.sum(lvt * w)) / 1e9) - ct_revenue_bn
        dividend = max(net_bn, 0.0) * 1e9 / total_people
        delta = ct - lvt + dividend * people
        bhc, ahc = poverty_rates(df, delta)
        rows.append(
            {
                "rate_pct": round(rate * 100, 2),
                "net_revenue_recycled_bn": round(max(net_bn, 0.0), 1),
                "dividend_per_person": round(dividend),
                "pct_winners": round(100 * float(np.sum((delta > 1) * w) / np.sum(w)), 1),
                "poverty_bhc_change": round(bhc - base_bhc, 2),
                "poverty_ahc_change": round(ahc - base_ahc, 2),
                "gini_change": round(income_gini(df, delta) - base_gini, 4),
                "decile_1": round(_wmean(delta[dec == 1], w[dec == 1])),
                "decile_10": round(_wmean(delta[dec == 10], w[dec == 10])),
            }
        )
    return {
        "design": "Net revenue above the council tax replacement returned as an equal per-person dividend",
        "baseline_poverty_bhc": round(base_bhc, 2),
        "baseline_poverty_ahc": round(base_ahc, 2),
        "scenarios": rows,
    }


def hvcts(df, rate):
    """High Value Council Tax Surcharge as a comparator instrument."""
    w = df["weight"].values
    val = df["main_residence_value"].values
    england = (df["country"].values == "ENGLAND")
    charge = np.zeros(len(df))
    for lo, hi, amount in HVCTS_BANDS:
        charge[(val >= lo) & (val < hi) & england] = amount
    liable = charge > 0
    lvt = rate * df["land"].values
    dec = df["income_decile"].values
    wd = df["wealth_decile"].values
    return {
        "design": "England only, 2026 values, flat bands (Autumn Budget 2025); simulated on 2026-27 microdata",
        "revenue_bn": round(float(np.sum(charge * w)) / 1e9, 2),
        "obr_costing_bn": 0.4,
        "properties_liable_thousands": round(float(np.sum(w[liable])) / 1e3),
        "obr_properties_thousands": 165,
        "share_of_households_pct": round(100 * float(np.sum(w[liable]) / np.sum(w)), 3),
        "avg_charge_if_liable": round(_wmean(charge[liable], w[liable])) if liable.any() else None,
        "share_of_lvt_revenue_pct": round(
            100 * float(np.sum(charge * w)) / float(np.sum(lvt * w)), 1
        ),
        "share_borne_by_top_wealth_decile_pct": round(
            100 * float(np.sum((charge * w)[wd == 10])) / max(float(np.sum(charge * w)), 1), 1
        ),
        "lvt_share_borne_by_top_wealth_decile_pct": round(
            100 * float(np.sum((lvt * w)[wd == 10])) / float(np.sum(lvt * w)), 1
        ),
        "share_borne_by_top_income_decile_pct": round(
            100 * float(np.sum((charge * w)[dec == 10])) / max(float(np.sum(charge * w)), 1), 1
        ),
        "avg_charge_top_wealth_decile": round(_wmean(charge[wd == 10], w[wd == 10])),
    }


def alternative_targets(df):
    """CT+SDLT replacement, and NI domestic rates folded in."""
    w = df["weight"].values
    land = df["land"].values
    ct = df["council_tax_saved"].values
    sdlt = df["sdlt"].values
    rates_ni = df["domestic_rates"].values
    land_bn = float(np.sum(land * w)) / 1e9
    dec = df["income_decile"].values
    wd = df["wealth_decile"].values

    def summary(abolished, label, note):
        target_bn = float(np.sum(abolished * w)) / 1e9
        r = target_bn / land_bn
        delta = abolished - r * land
        bhc, ahc = poverty_rates(df, delta)
        pw = df["person_weight"].values
        base_bhc = 100 * float(np.sum(df["in_poverty_bhc"].values * pw) / np.sum(pw))
        base_ahc = 100 * float(np.sum(df["in_poverty_ahc"].values * pw) / np.sum(pw))
        return {
            "label": label,
            "note": note,
            "revenue_replaced_bn": round(target_bn, 1),
            "rate_pct": round(r * 100, 3),
            "pct_winners": round(100 * float(np.sum((delta > 1) * w) / np.sum(w)), 1),
            "decile_1": round(_wmean(delta[dec == 1], w[dec == 1])),
            "decile_10": round(_wmean(delta[dec == 10], w[dec == 10])),
            "wealth_decile_10": round(_wmean(delta[wd == 10], w[wd == 10])),
            "poverty_bhc_change": round(bhc - base_bhc, 2),
            "poverty_ahc_change": round(ahc - base_ahc, 2),
        }

    return {
        "sdlt_revenue_bn": round(float(np.sum(sdlt * w)) / 1e9, 1),
        "ni_domestic_rates_bn": round(float(np.sum(rates_ni * w)) / 1e9, 2),
        "scenarios": [
            summary(ct, "Council tax only (central)", "as in the main results"),
            summary(
                ct + sdlt,
                "Council tax and stamp duty",
                "the Mirrlees / Leunig / Muellbauer proposal",
            ),
            summary(
                ct + rates_ni,
                "Council tax and NI domestic rates",
                "comprehensive UK-wide replacement",
            ),
            summary(
                ct + sdlt + rates_ni,
                "Council tax, stamp duty and NI rates",
                "all recurrent and transaction property taxes on dwellings",
            ),
        ],
    }


def exempt_band(df, ct_revenue_bn):
    w = df["weight"].values
    land = df["land"].values
    ct = df["council_tax_saved"].values
    dec = df["income_decile"].values
    wd = df["wealth_decile"].values
    pw = df["person_weight"].values
    base_bhc = 100 * float(np.sum(df["in_poverty_bhc"].values * pw) / np.sum(pw))
    base_ahc = 100 * float(np.sum(df["in_poverty_ahc"].values * pw) / np.sum(pw))
    rows = []
    for allowance in (0,) + EXEMPT_BANDS:
        base = np.maximum(land - allowance, 0)
        base_bn = float(np.sum(base * w)) / 1e9
        r = ct_revenue_bn / base_bn
        delta = ct - r * base
        bhc, ahc = poverty_rates(df, delta)
        rows.append(
            {
                "exempt_band": allowance,
                "taxable_base_tn": round(base_bn / 1e3, 3),
                "rate_pct": round(r * 100, 3),
                "pct_winners": round(100 * float(np.sum((delta > 1) * w) / np.sum(w)), 1),
                "pct_untaxed_households": round(
                    100 * float(np.sum(w[base <= 0]) / np.sum(w)), 1
                ),
                "decile_1": round(_wmean(delta[dec == 1], w[dec == 1])),
                "decile_10": round(_wmean(delta[dec == 10], w[dec == 10])),
                "wealth_decile_10": round(_wmean(delta[wd == 10], w[wd == 10])),
                "poverty_bhc_change": round(bhc - base_bhc, 2),
                "poverty_ahc_change": round(ahc - base_ahc, 2),
            }
        )
    return rows


def build():
    sim, df = load()
    w = df["weight"].values
    ct_revenue_bn = float(np.sum(df["council_tax_saved"].values * w)) / 1e9
    land_bn = float(np.sum(df["land"].values * w)) / 1e9
    rate = ct_revenue_bn / land_bn

    return {
        "budget_neutral_rate_pct": round(rate * 100, 3),
        "etr_by_property_value": etr_by_property_value(df, rate),
        "etr_by_main_residence_value": etr_by_property_value(
            df, rate, value_col="main_residence_value"
        ),
        "top_tail_reconciliation": top_tail_reconciliation(df),
        "progressivity": progressivity(df, rate),
        "by_tenure": by_tenure(df, rate),
        "decile_uncertainty": decile_uncertainty(df, rate),
        "losers": losers(df, rate),
        "recycling": recycling(df, ct_revenue_bn, rate),
        "hvcts": hvcts(df, rate),
        "alternative_targets": alternative_targets(df),
        "exempt_band": exempt_band(df, ct_revenue_bn),
    }


def main():
    res = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
