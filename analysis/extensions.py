"""Robustness, decomposition and extension runs for the LVT paper.

Runs directly against ``policyengine_uk.Microsimulation`` on the same pinned
Enhanced FRS release used by ``uk_lvt/pipeline.py``, and writes
``results/extensions.json``.

Contents
--------
``validation``
    Checks that the household net income change under the swap is exactly
    ``council tax saved - LVT`` (i.e. that there are no benefit interactions),
    and that an arithmetic replication of the model's poverty statistics
    reproduces the model's own reform run. Every robustness variant below is
    computed with the validated arithmetic rather than a fresh solver run.

``land_decomposition``
    Household land, allocated corporate land and property wealth by income
    decile, so that the two definitions of "land" are never mixed in a table.

``robustness``
    Revenue-neutral rate and distributional summary under alternative
    replacement targets (OBR receipts), tax bases (household-only land,
    foreign-owned corporate equity, ONS-scaled household land, land-share
    perturbations), geographic scope (Great Britain only) and instrument
    (proportional property tax).

``capitalisation``
    Arithmetic capitalisation of the permanent levy into land prices at a
    range of discount rates, by wealth decile.

``single_household``
    Worked example computed against the population simulation rather than a
    one-household simulation (which mis-allocates the entire corporate land
    base to the single record).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = "hf://policyengine/policyengine-uk-data-private/enhanced_frs_2023_24.h5@1.55.10"
YEAR = 2026
OUTPUT = ROOT / "results" / "extensions.json"

# Replacement targets (£bn).
CT_MODEL_BN = None  # filled from the simulation
CT_OBR_BN = 53.7  # OBR March 2025 EFO accruals-based council tax receipts

DISCOUNT_RATES = (0.03, 0.04, 0.05)
FOREIGN_EQUITY_SHARE = 0.5  # share of UK corporate equity held abroad


def _v(sim, name, year=YEAR):
    return np.asarray(sim.calculate(name, year).values, dtype=np.float64)


def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _gini(x, w):
    """Weighted Gini coefficient."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    total = cxw[-1]
    if total == 0:
        return 0.0
    return float(1 - np.sum((cxw + np.concatenate([[0], cxw[:-1]])) * w) / (total * cw[-1]))


def _deciles(values, weights, n=10):
    """Weighted decile index (1-10) of ``values``."""
    order = np.argsort(values)
    cw = np.cumsum(weights[order]) / np.sum(weights)
    d = np.searchsorted(np.arange(1, n) / n, cw, side="left") + 1
    out = np.empty(len(values), dtype=int)
    out[order] = np.minimum(d, n)
    return out


def load_baseline():
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=DATASET)
    cols = {
        "weight": _v(sim, "household_weight"),
        "net_income": _v(sim, "household_net_income"),
        "council_tax": _v(sim, "council_tax"),
        "council_tax_less_benefit": _v(sim, "council_tax_less_benefit"),
        "household_land": _v(sim, "household_land_value"),
        "corporate_land": _v(sim, "corporate_land_value"),
        "land": _v(sim, "land_value"),
        "property_wealth": _v(sim, "property_wealth"),
        "total_wealth": _v(sim, "total_wealth"),
        "income_decile": _v(sim, "household_income_decile"),
        "wealth_decile": _v(sim, "household_wealth_decile"),
        "in_poverty_bhc": _v(sim, "in_poverty_bhc"),
        "in_poverty_ahc": _v(sim, "in_poverty_ahc"),
        "equiv_hbai_bhc": _v(sim, "equiv_hbai_household_net_income"),
        "equiv_hbai_ahc": _v(sim, "equiv_hbai_household_net_income_ahc"),
        "equivalisation_bhc": _v(sim, "household_equivalisation_bhc"),
        "equivalisation_ahc": _v(sim, "household_equivalisation_ahc"),
        "poverty_line_bhc": _v(sim, "poverty_line_bhc"),
        "poverty_line_ahc": _v(sim, "poverty_line_ahc"),
    }
    df = pd.DataFrame(cols)
    df["country"] = np.asarray(sim.calculate("country", YEAR).values).astype(str)
    # The model's own decile variables are degenerate at the bottom of the
    # wealth distribution (large mass of tied zero-wealth households), so we
    # also construct equal-weight deciles and use those for the tables.
    df["income_decile_own"] = _deciles(df["net_income"].values, df["weight"].values)
    df["wealth_decile_own"] = _deciles(df["total_wealth"].values, df["weight"].values)
    return sim, df


def reform_sim(rate, abolish_ct=True):
    from policyengine_uk import Microsimulation

    reform = {
        "gov.contrib.ubi_center.land_value_tax.rate": {
            f"{YEAR}-01-01.{YEAR + 4}-12-31": rate
        }
    }
    if abolish_ct:
        reform["gov.contrib.abolish_council_tax"] = {
            f"{YEAR}-01-01.{YEAR + 4}-12-31": True
        }
    return Microsimulation(dataset=DATASET, reform=reform)


# ---------------------------------------------------------------------------
# Arithmetic replication of the swap
# ---------------------------------------------------------------------------


def poverty_from_change(df, delta):
    """Poverty rates (%) under a net income change ``delta``, baseline lines."""
    bhc = df["equiv_hbai_bhc"] + delta / df["equivalisation_bhc"]
    ahc = df["equiv_hbai_ahc"] + delta / df["equivalisation_ahc"]
    w = df["weight"].values
    line_bhc = df["poverty_line_bhc"] / df["equivalisation_bhc"]
    line_ahc = df["poverty_line_ahc"] / df["equivalisation_ahc"]
    return (
        100 * float(np.sum((bhc.values < line_bhc.values) * w) / np.sum(w)),
        100 * float(np.sum((ahc.values < line_ahc.values) * w) / np.sum(w)),
    )


def scenario(df, base, ct_saved, revenue_target_bn, label, note=""):
    """Distributional summary of replacing ``ct_saved`` with an LVT on ``base``."""
    w = df["weight"].values
    base_total_bn = float(np.sum(base * w)) / 1e9
    rate = revenue_target_bn / base_total_bn
    lvt = rate * base
    delta = ct_saved - lvt

    net = df["net_income"].values
    gini_base = _gini(net, w)
    gini_reform = _gini(net + delta, w)
    pov_bhc, pov_ahc = poverty_from_change(df, delta)
    pov_bhc_base = 100 * float(np.sum(df["in_poverty_bhc"].values * w) / np.sum(w))
    pov_ahc_base = 100 * float(np.sum(df["in_poverty_ahc"].values * w) / np.sum(w))

    inc_d = df["income_decile"].values  # model equivalised-income deciles
    wealth_d = df["wealth_decile_own"].values

    def dec_mean(d, k):
        m = d == k
        if not m.any():
            return float("nan")
        return _wmean(delta[m], w[m])

    def r0(x):
        return None if np.isnan(x) else round(x)

    return {
        "label": label,
        "note": note,
        "base_tn": round(base_total_bn / 1e3, 3),
        "revenue_target_bn": round(revenue_target_bn, 1),
        "rate_pct": round(rate * 100, 3),
        "pct_winners": round(100 * float(np.sum((delta > 1) * w) / np.sum(w)), 1),
        "pct_losers": round(100 * float(np.sum((delta < -1) * w) / np.sum(w)), 1),
        "aggregate_change_bn": round(float(np.sum(delta * w)) / 1e9, 2),
        "income_decile_1": r0(dec_mean(inc_d, 1)),
        "income_decile_5": r0(dec_mean(inc_d, 5)),
        "income_decile_9": r0(dec_mean(inc_d, 9)),
        "income_decile_10": r0(dec_mean(inc_d, 10)),
        "wealth_decile_1": r0(dec_mean(wealth_d, 1)),
        "wealth_decile_10": r0(dec_mean(wealth_d, 10)),
        "poverty_bhc_change": round(pov_bhc - pov_bhc_base, 2),
        "poverty_ahc_change": round(pov_ahc - pov_ahc_base, 2),
        "gini_change": round(gini_reform - gini_base, 4),
    }


def build() -> dict:
    sim, df = load_baseline()
    w = df["weight"].values
    ct_saved = df["council_tax_less_benefit"].values
    ct_model_bn = float(np.sum(ct_saved * w)) / 1e9
    land = df["land"].values
    hh_land = df["household_land"].values
    corp_land = df["corporate_land"].values

    out: dict = {}

    # Integrity check on the model's own decile variables: the wealth deciles
    # are degenerate because of the mass of tied zero-wealth households.
    def group_shares(col):
        s = df.groupby(col)["weight"].sum() / df["weight"].sum() * 100
        return {int(k): round(float(v), 1) for k, v in s.items()}

    out["decile_integrity"] = {
        "model_income_decile_population_share_pct": group_shares("income_decile"),
        "model_wealth_decile_population_share_pct": group_shares("wealth_decile"),
        "reconstructed_wealth_decile_population_share_pct": group_shares("wealth_decile_own"),
        "note": (
            "Model wealth deciles omit decile 1 entirely and over-fill decile 2 "
            "because households with zero or negative net wealth are tied; "
            "reconstructed deciles are equal-weight."
        ),
    }

    out["aggregates"] = {
        "households_m": round(float(np.sum(w)) / 1e6, 2),
        "council_tax_gross_bn": round(float(np.sum(df["council_tax"].values * w)) / 1e9, 1),
        "council_tax_net_bn": round(ct_model_bn, 1),
        "council_tax_obr_bn": CT_OBR_BN,
        "household_land_tn": round(float(np.sum(hh_land * w)) / 1e12, 3),
        "corporate_land_tn": round(float(np.sum(corp_land * w)) / 1e12, 3),
        "total_land_tn": round(float(np.sum(land * w)) / 1e12, 3),
        "property_wealth_tn": round(float(np.sum(df["property_wealth"].values * w)) / 1e12, 3),
    }

    # -- validation ---------------------------------------------------------
    central_rate = ct_model_bn / (float(np.sum(land * w)) / 1e9)
    rsim = reform_sim(central_rate)
    model_delta = _v(rsim, "household_net_income") - df["net_income"].values
    arithmetic_delta = ct_saved - central_rate * land
    model_pov_bhc = 100 * float(
        np.sum(_v(rsim, "in_poverty_bhc") * w) / np.sum(w)
    )
    model_pov_ahc = 100 * float(
        np.sum(_v(rsim, "in_poverty_ahc") * w) / np.sum(w)
    )
    arith_pov_bhc, arith_pov_ahc = poverty_from_change(df, arithmetic_delta)

    out["validation"] = {
        "central_rate_pct": round(central_rate * 100, 3),
        "max_abs_delta_gap": round(float(np.max(np.abs(model_delta - arithmetic_delta))), 2),
        "mean_abs_delta_gap": round(float(np.mean(np.abs(model_delta - arithmetic_delta))), 4),
        "share_households_with_benefit_interaction": round(
            100
            * float(np.sum((np.abs(model_delta - arithmetic_delta) > 1) * w) / np.sum(w)),
            3,
        ),
        "model_poverty_bhc": round(model_pov_bhc, 2),
        "arithmetic_poverty_bhc": round(arith_pov_bhc, 2),
        "model_poverty_ahc": round(model_pov_ahc, 2),
        "arithmetic_poverty_ahc": round(arith_pov_ahc, 2),
    }

    # -- land decomposition by income decile --------------------------------
    rows = []
    for d in range(1, 11):
        m = df["income_decile"].values == d
        rows.append(
            {
                "decile": d,
                "avg_household_land": round(_wmean(hh_land[m], w[m])),
                "avg_corporate_land": round(_wmean(corp_land[m], w[m])),
                "avg_total_land": round(_wmean(land[m], w[m])),
                "avg_property_wealth": round(_wmean(df["property_wealth"].values[m], w[m])),
                "household_land_share_of_property_pct": round(
                    100
                    * float(np.sum(hh_land[m] * w[m]))
                    / float(np.sum(df["property_wealth"].values[m] * w[m])),
                    1,
                ),
                "share_of_total_land_pct": round(
                    100 * float(np.sum(land[m] * w[m])) / float(np.sum(land * w)), 1
                ),
            }
        )
    out["land_decomposition"] = rows

    # -- robustness ---------------------------------------------------------
    gb = (df["country"].values != "NORTHERN_IRELAND").astype(float)
    scenarios = [
        scenario(df, land, ct_saved, ct_model_bn, "Central", "0.77% on all land"),
        scenario(
            df, land, ct_saved, CT_OBR_BN, "OBR receipts target",
            "replacement target set to OBR council tax receipts",
        ),
        scenario(
            df, land * gb, ct_saved, ct_model_bn, "Great Britain only",
            "Northern Ireland excluded from the LVT base",
        ),
        scenario(
            df, hh_land, ct_saved, ct_model_bn, "Household land only",
            "corporate land excluded from the base",
        ),
        scenario(
            df, hh_land + (1 - FOREIGN_EQUITY_SHARE) * corp_land, ct_saved, ct_model_bn,
            "Foreign-owned corporate equity",
            "half of corporate land assumed foreign-owned and outside the household base",
        ),
        scenario(
            df, hh_land / 1.0712 + corp_land, ct_saved, ct_model_bn,
            "Household land scaled to ONS 2024",
            "household land rescaled to the un-uprated ONS benchmark",
        ),
        scenario(
            df, 0.9 * hh_land + corp_land, ct_saved, ct_model_bn,
            "Land shares -10%", "all regional land shares scaled by 0.9",
        ),
        scenario(
            df, 1.1 * hh_land + corp_land, ct_saved, ct_model_bn,
            "Land shares +10%", "all regional land shares scaled by 1.1",
        ),
        scenario(
            df, df["property_wealth"].values, ct_saved, ct_model_bn,
            "Proportional property tax",
            "comparator: flat tax on total property value, structures included",
        ),
    ]
    out["robustness"] = scenarios

    # -- capitalisation -----------------------------------------------------
    cap_rows = []
    for rho in DISCOUNT_RATES:
        drop = central_rate / rho
        per_decile = []
        for d in range(1, 11):
            m = df["wealth_decile_own"].values == d
            loss = drop * _wmean(land[m], w[m])
            wealth = _wmean(df["total_wealth"].values[m], w[m])
            per_decile.append(
                {
                    "decile": d,
                    "capitalised_land_loss": round(loss),
                    "pct_of_total_wealth": round(100 * loss / wealth, 1) if wealth else None,
                }
            )
        cap_rows.append(
            {
                "discount_rate": rho,
                "land_price_fall_pct": round(100 * drop, 1),
                "aggregate_loss_tn": round(drop * float(np.sum(land * w)) / 1e12, 2),
                "by_wealth_decile": per_decile,
            }
        )
    out["capitalisation"] = {"rate_pct": round(central_rate * 100, 3), "scenarios": cap_rows}

    # -- pensioner deferral -------------------------------------------------
    from policyengine_uk import Microsimulation  # noqa: F401

    sp = np.asarray(sim.calculate("is_SP_age", YEAR).values, dtype=float)
    hh_of_person = np.asarray(
        sim.calculate("household_id", YEAR, map_to="person").values
    )
    hh_ids = np.asarray(sim.calculate("household_id", YEAR).values)
    pensioner_hh = (
        pd.Series(sp).groupby(pd.Series(hh_of_person)).max().reindex(hh_ids).fillna(0).values
    )
    lvt_central = central_rate * land
    deferred = float(np.sum(lvt_central * pensioner_hh * w)) / 1e9
    out["pensioner_deferral"] = {
        "pensioner_household_share_pct": round(
            100 * float(np.sum(pensioner_hh * w) / np.sum(w)), 1
        ),
        "lvt_deferred_bn": round(deferred, 1),
        "share_of_lvt_deferred_pct": round(
            100 * deferred / (float(np.sum(lvt_central * w)) / 1e9), 1
        ),
        "avg_deferred_liability": round(
            _wmean(lvt_central[pensioner_hh > 0], w[pensioner_hh > 0])
        ),
    }

    # -- worked single household --------------------------------------------
    corp_per_pound = float(np.sum(corp_land * w)) / float(
        np.sum(_v(sim, "corporate_wealth") * w)
    )
    out["single_household"] = {
        "property_value": 400_000,
        "land_share_london": 0.85,
        "household_land_value": 340_000,
        "corporate_wealth": 50_000,
        "corporate_land_per_pound_of_corporate_wealth": round(corp_per_pound, 3),
        "allocated_corporate_land": round(50_000 * corp_per_pound),
        "total_land_value": round(340_000 + 50_000 * corp_per_pound),
        "lvt_at_central_rate": round(central_rate * (340_000 + 50_000 * corp_per_pound)),
    }

    return out


def main() -> None:
    results = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")
    print(json.dumps(results["validation"], indent=2))


if __name__ == "__main__":
    main()
