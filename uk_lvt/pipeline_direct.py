"""Regenerate ``results/lvt_results.json`` from a single baseline simulation.

Why this exists
---------------
``uk_lvt/pipeline.py`` drives the policyengine.py (v4) client and issues one
solver run per scenario. This module produces the same results file from a
single baseline run of ``policyengine_uk.Microsimulation`` plus closed-form
arithmetic, which is both faster and independently checkable.

The arithmetic is exact, not an approximation, and the module verifies it on
every invocation:

* the land value tax is linear in the rate, ``LVT_i = r L_i``, because the base
  is a stock that the reform does not alter;
* the swap generates no benefit interactions, so the change in household net
  income is exactly ``CT_i - r L_i`` up to the model's float32 rounding
  (checked against full reform runs: every household's gap must be within
  ``FLOAT32_ULP_TOLERANCE`` float32 spacings of its largest income, benefit
  or tax total, see
  ``uk_lvt/validation.py``; the share of households for which this fails is
  reported as ``validation.share_households_with_benefit_interaction``);
* poverty under the reform can therefore be recomputed by shifting equivalised
  HBAI income by that change and comparing with the baseline-fixed thresholds
  (checked against the model's own ``in_poverty_bhc`` / ``in_poverty_ahc`` under
  full reform runs at three rates).

If any check fails the module raises rather than writing a results file.

Reproducing
-----------
``uv sync --frozen --extra simulation`` installs the pinned model, then
``uv run python -m uk_lvt.pipeline_direct`` regenerates the file. Pass
``--dataset`` (or set ``UK_LVT_DATASET``) to read a local copy of the pinned
dataset, and ``--uk-data-root`` (or ``POLICYENGINE_UK_DATA_ROOT``) for the
ONS land targets; ``scripts/fetch_uk_data_targets.py`` fetches those at the
pinned policyengine-uk-data commit. The output records the package versions
and the dataset SHA-256 in its ``provenance`` block.

Deciles
-------
Income deciles are the model's own ``household_income_decile`` (deciles of
equivalised net income). Wealth deciles are reconstructed as equal-weight
deciles of total wealth: the model's ``household_wealth_decile`` leaves decile
one empty and places 22.6 per cent of households in decile two, because
households with zero or negative net wealth are tied.

Statistical conventions
-----------------------
Poverty rates are shares of *individuals* (household poverty status weighted
by household weight times household size), matching the HBAI convention. The
income Gini is computed on *equivalised* HBAI household net income (BHC),
person-weighted. The wealth Gini is computed on total household wealth,
household-weighted, matching the ONS WAS convention.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    DEFAULT_LVT_RATES,
    build_average_land_tables,
    build_baseline_summary,
    build_council_tax_vs_lvt_table,
    build_distribution_by_decile,
    build_impact_scenario_table,
    build_landless_summary,
    build_ons_comparison,
    build_regional_impact_table,
    build_revenue_by_rate,
    build_revenue_by_scope,
    classify_family_type,
    format_rate_label,
    make_rate_grid,
)
from .provenance import build_provenance, dataset_source
from .validation import (
    FLOAT32_ULP_TOLERANCE,
    REPORTING_THRESHOLD_GBP,
    budget_neutral_rate,
    closed_form_gap_check,
    household_scale,
)

DATASET = dataset_source()
YEAR = 2026
OUTPUT = Path("results/lvt_results.json")
VALIDATION_RATES = (0.005, 0.01, 0.02)
# Reporting threshold for share_households_gap_over_threshold_gbp. The guard
# itself is relative: FLOAT32_ULP_TOLERANCE float32 spacings of each
# household's largest total (uk_lvt/validation.py).
TOLERANCE_GBP = REPORTING_THRESHOLD_GBP
TOLERANCE_PP = 0.01
# household_net_income = market income + benefits - tax - pension
# contributions. The largest absolute value of net income, market income,
# benefits and tax, in the baseline or reform run, sets each household's
# float32 rounding scale.
NET_INCOME_TOTALS = (
    "household_net_income",
    "household_market_income",
    "household_benefits",
    "household_tax",
)


def _v(sim, name, year=YEAR, dtype=np.float64):
    return np.asarray(sim.calculate(name, year).values, dtype=dtype)


def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _gini(x, w):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    if cxw[-1] == 0:
        return 0.0
    lorenz = cxw + np.concatenate([[0.0], cxw[:-1]])
    return float(1 - np.sum(lorenz * w) / (cxw[-1] * cw[-1]))


def equal_weight_deciles(values, weights, n=10):
    order = np.argsort(values, kind="stable")
    cw = np.cumsum(weights[order]) / np.sum(weights)
    d = np.searchsorted(np.arange(1, n) / n, cw, side="left") + 1
    out = np.empty(len(values), dtype=int)
    out[order] = np.minimum(d, n)
    return out


def load_baseline():
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=DATASET)
    df = pd.DataFrame(
        {
            "weight": _v(sim, "household_weight"),
            "income": _v(sim, "household_net_income"),
            "council_tax": _v(sim, "council_tax"),
            "council_tax_benefit": np.asarray(
                sim.calculate("council_tax_benefit", YEAR, map_to="household").values,
                dtype=np.float64,
            ),
            "hh_land": _v(sim, "household_land_value"),
            "corp_land": _v(sim, "corporate_land_value"),
            "land_value": _v(sim, "land_value"),
            "property_wealth": _v(sim, "property_wealth"),
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
    for name in NET_INCOME_TOTALS[1:]:
        df[name] = _v(sim, name)
    df["council_tax_saved"] = (
        df["council_tax"] - df["council_tax_benefit"]
    )
    df["country"] = np.asarray(sim.calculate("country", YEAR).values).astype(str)
    df["region"] = np.asarray(sim.calculate("region", YEAR).values).astype(str)
    df["wealth_decile"] = equal_weight_deciles(
        df["total_wealth"].values, df["weight"].values
    )
    hh_of_person = np.asarray(sim.calculate("household_id", YEAR, map_to="person").values)
    ids = np.asarray(sim.calculate("household_id", YEAR).values)
    df["people"] = (
        pd.Series(1.0, index=hh_of_person).groupby(level=0).sum().reindex(ids).fillna(0).values
    )
    # Person weight: poverty rates are shares of individuals (HBAI convention)
    # and the income Gini is person-weighted over equivalised income.
    df["person_weight"] = df["weight"] * df["people"]
    return sim, df


def reform_run(rate, abolish_ct=True):
    from policyengine_uk import Microsimulation

    window = f"{YEAR}-01-01.{YEAR + 4}-12-31"
    reform = {"gov.contrib.ubi_center.land_value_tax.rate": {window: rate}}
    if abolish_ct:
        reform["gov.contrib.abolish_council_tax"] = {window: True}
    return Microsimulation(dataset=DATASET, reform=reform)


def poverty_rates(df, delta):
    """Poverty rates (% of individuals) under net income change ``delta``.

    Household poverty status is weighted by household weight times household
    size, giving the share of people in poor households (HBAI convention),
    against baseline-fixed thresholds.
    """
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
    return _gini(equiv, df["person_weight"].values)


def validate(df) -> dict:
    """Check linearity, absence of benefit interactions, and poverty replication.

    A household's full-model change in net income must equal the closed form
    ``CT_i - r L_i`` to within ``FLOAT32_ULP_TOLERANCE`` float32 spacings of
    its largest total among ``NET_INCOME_TOTALS`` (baseline or reform run). The model computes these totals in float32, so the rounding
    gap grows with the household's income; an absolute £ tolerance would trip
    on very high-income households whose float32 spacing approaches £1.
    """
    w = df["weight"].values
    pw = df["person_weight"].values
    land = df["land_value"].values
    ct = df["council_tax_saved"].values
    baseline_totals = [df["income"].values] + [
        df[name].values for name in NET_INCOME_TOTALS[1:]
    ]
    checks = []
    for rate in VALIDATION_RATES:
        sim = reform_run(rate)
        model_lvt = _v(sim, "LVT")
        reform_totals = [_v(sim, name) for name in NET_INCOME_TOTALS]
        model_delta = reform_totals[0] - df["income"].values
        arith_delta = ct - rate * land
        gap = closed_form_gap_check(
            model_delta,
            arith_delta,
            household_scale(*baseline_totals, *reform_totals),
            w,
            ulps=FLOAT32_ULP_TOLERANCE,
            reporting_threshold_gbp=TOLERANCE_GBP,
        )
        pov_bhc, pov_ahc = poverty_rates(df, arith_delta)
        model_bhc = 100 * float(np.sum(_v(sim, "in_poverty_bhc") * pw) / np.sum(pw))
        model_ahc = 100 * float(np.sum(_v(sim, "in_poverty_ahc") * pw) / np.sum(pw))
        check = {
            "rate": rate,
            "max_abs_lvt_gap": float(np.max(np.abs(model_lvt - rate * land))),
            "max_abs_delta_gap": gap["max_abs_delta_gap"],
            "share_households_with_benefit_interaction": gap[
                "share_households_with_benefit_interaction"
            ],
            "model_poverty_bhc": model_bhc,
            "arithmetic_poverty_bhc": pov_bhc,
            "model_poverty_ahc": model_ahc,
            "arithmetic_poverty_ahc": pov_ahc,
            "max_delta_gap_float32_ulps": gap["max_delta_gap_float32_ulps"],
            "share_households_gap_over_threshold_gbp": gap[
                "share_households_gap_over_threshold_gbp"
            ],
        }
        if not gap["passed"]:
            raise RuntimeError(
                f"Net income change is not CT - rL within {FLOAT32_ULP_TOLERANCE} "
                f"float32 spacings at rate {rate}: {check}"
            )
        if abs(model_bhc - pov_bhc) > TOLERANCE_PP or abs(model_ahc - pov_ahc) > TOLERANCE_PP:
            raise RuntimeError(f"Poverty replication failed at rate {rate}: {check}")
        checks.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in check.items()})
    return {
        "tolerance_gbp": TOLERANCE_GBP,
        "tolerance_pp": TOLERANCE_PP,
        "tolerance_float32_ulps": FLOAT32_ULP_TOLERANCE,
        "checks": checks,
    }


def _family_types(sim, n_households: int) -> list[str]:
    age = _v(sim, "age", dtype=float)
    sp = _v(sim, "is_SP_age", dtype=float)
    hh = np.asarray(sim.calculate("household_id", YEAR, map_to="person").values)
    ids = np.asarray(sim.calculate("household_id", YEAR).values)
    grouped = (
        pd.DataFrame(
            {
                "household_id": hh,
                "children": (age < 18).astype(float),
                "adults": (age >= 18).astype(float),
                "pensioners": sp,
            }
        )
        .groupby("household_id")
        .sum()
        .reindex(ids)
        .fillna(0)
    )
    return [
        classify_family_type(a, c, p)
        for a, c, p in zip(
            grouped["adults"].values, grouped["children"].values, grouped["pensioners"].values
        )
    ]


def build_results(uk_data_root: Path | None = None) -> dict:
    from .pipeline import _load_ons_land_targets

    sim, df = load_baseline()
    w = df["weight"].values
    land = df["land_value"].values
    ct = df["council_tax_saved"].values

    results: dict = {
        "model_version": _model_version(),
        "dataset": DATASET,
        "provenance": build_provenance(DATASET),
    }
    results["validation"] = validate(df)

    baseline_df = df.rename(columns={"income": "income"})[
        [
            "land_value",
            "hh_land",
            "corp_land",
            "property_wealth",
            "total_wealth",
            "income",
            "income_decile",
            "wealth_decile",
            "weight",
        ]
    ]
    results["baseline"] = build_baseline_summary(baseline_df)

    household_df = pd.DataFrame(
        {
            "land_value": land,
            "country": df["country"].values,
            "region": df["region"].values,
            "family_type": _family_types(sim, len(df)),
            "weight": w,
        }
    )
    (
        results["avg_land_by_country"],
        results["avg_land_by_region"],
        results["avg_land_by_family_type"],
    ) = build_average_land_tables(household_df)

    results["ons_comparison"] = build_ons_comparison(
        results["baseline"], **_load_ons_land_targets(uk_data_root=uk_data_root)
    )
    results["distribution_by_decile"] = build_distribution_by_decile(baseline_df)
    results["distribution_by_wealth_decile"] = build_distribution_by_decile(
        baseline_df, decile_col="wealth_decile"
    )

    required_rate, council_tax_revenue_bn, total_land_bn = budget_neutral_rate(
        ct, land, w
    )

    results["revenue_by_rate"] = build_revenue_by_rate(
        council_tax_revenue_bn,
        [
            {
                "rate": rate,
                "lvt_revenue_bn": rate * total_land_bn,
                "avg_per_household": rate * _wmean(land, w),
            }
            for rate in DEFAULT_LVT_RATES
        ],
    )

    pw = df["person_weight"].values
    baseline_bhc = 100 * float(np.sum(df["in_poverty_bhc"].values * pw) / np.sum(pw))
    baseline_ahc = 100 * float(np.sum(df["in_poverty_ahc"].values * pw) / np.sum(pw))
    baseline_gini = income_gini(df)
    baseline_wealth_gini = _gini(df["total_wealth"].values, w)

    results["impact_scenarios"] = {}
    results["impact_scenarios_by_wealth"] = {}
    results["impact_by_region"] = []
    results["landless_summary"] = {}
    results["council_tax_vs_lvt_scenarios"] = {}
    results["council_tax_vs_lvt_scenarios_by_wealth"] = {}
    results["poverty_gini"] = {
        "baseline_poverty_bhc": round(baseline_bhc, 2),
        "baseline_poverty_ahc": round(baseline_ahc, 2),
        "baseline_gini": round(baseline_gini, 4),
        "baseline_wealth_gini": round(baseline_wealth_gini, 4),
        "scenarios": {},
    }

    for rate in make_rate_grid(required_rate):
        label = format_rate_label(rate, required_rate)
        lvt = rate * land
        delta = ct - lvt
        pov_bhc, pov_ahc = poverty_rates(df, delta)
        gini = income_gini(df, delta)
        results["poverty_gini"]["scenarios"][label] = {
            "poverty_bhc": round(pov_bhc, 2),
            "poverty_ahc": round(pov_ahc, 2),
            "poverty_bhc_change": round(pov_bhc - baseline_bhc, 2),
            "poverty_ahc_change": round(pov_ahc - baseline_ahc, 2),
            "gini": round(gini, 4),
            "gini_change": round(gini - baseline_gini, 4),
            "wealth_gini": round(baseline_wealth_gini, 4),
            "wealth_gini_change": 0.0,
        }

        impact_df = pd.DataFrame(
            {
                "income_decile": df["income_decile"].values,
                "wealth_decile": df["wealth_decile"].values,
                "lvt": lvt,
                "council_tax_saved": ct,
                "income_change": delta,
                "baseline_income": df["income"].values,
                "land_value": land,
                "region": df["region"].values,
                "weight": w,
            }
        )
        results["impact_scenarios"][label] = build_impact_scenario_table(impact_df)
        results["impact_scenarios_by_wealth"][label] = build_impact_scenario_table(
            impact_df, decile_col="wealth_decile"
        )
        if np.isclose(rate, required_rate, atol=1e-6):
            results["impact_by_region"] = build_regional_impact_table(impact_df)
        results["landless_summary"][label] = build_landless_summary(impact_df)

        ct_vs_lvt = pd.DataFrame(
            {
                "income_decile": df["income_decile"].values,
                "wealth_decile": df["wealth_decile"].values,
                "council_tax": ct,
                "lvt": lvt,
                "weight": w,
            }
        )
        results["council_tax_vs_lvt_scenarios"][label] = build_council_tax_vs_lvt_table(
            ct_vs_lvt
        )
        results["council_tax_vs_lvt_scenarios_by_wealth"][label] = (
            build_council_tax_vs_lvt_table(ct_vs_lvt, decile_col="wealth_decile")
        )

    hh_land_bn = float(np.sum(df["hh_land"].values * w)) / 1e9
    corp_land_bn = float(np.sum(df["corp_land"].values * w)) / 1e9
    results["revenue_by_scope"] = build_revenue_by_scope(
        [
            {
                "scope": "all_land",
                "revenue_bn": 0.01 * total_land_bn,
                "avg_per_household": 0.01 * _wmean(land, w),
            },
            {
                "scope": "household_only",
                "revenue_bn": 0.01 * hh_land_bn,
                "avg_per_household": 0.01 * _wmean(df["hh_land"].values, w),
            },
            {
                "scope": "corporate_only",
                "revenue_bn": 0.01 * corp_land_bn,
                "avg_per_household": 0.01 * _wmean(df["corp_land"].values, w),
            },
        ]
    )

    results["council_tax_replacement"] = {
        "council_tax_revenue_bn": round(council_tax_revenue_bn, 1),
        "council_tax_gross_bn": round(float(np.sum(df["council_tax"].values * w)) / 1e9, 1),
        "council_tax_benefit_bn": round(
            float(np.sum(df["council_tax_benefit"].values * w)) / 1e9, 1
        ),
        "total_land_bn": round(total_land_bn, 1),
        "required_lvt_rate_pct": round(required_rate * 100, 2),
        "households_m": round(float(np.sum(w)) / 1e6, 2),
    }

    # Worked single-household example, computed against population aggregates
    # rather than a one-household simulation.
    corp_land_per_pound = corp_land_bn / (
        float(np.sum(_v(sim, "corporate_wealth") * w)) / 1e9
    )
    allocated = 50_000 * corp_land_per_pound
    results["single_household_example"] = {
        "property_value": 400_000,
        "land_share": 0.85,
        "household_land_value": 340_000,
        "corporate_wealth": 50_000,
        "corporate_land_per_pound_of_corporate_wealth": round(corp_land_per_pound, 3),
        "corporate_land_value": round(allocated),
        "total_land_value": round(340_000 + allocated),
        "lvt_liability_budget_neutral": round(required_rate * (340_000 + allocated)),
        "lvt_liability_1pct": round(0.01 * (340_000 + allocated)),
    }

    return results


def _model_version() -> str:
    import importlib.metadata as md

    version = os.environ.get("POLICYENGINE_UK_VERSION", md.version("policyengine-uk"))
    commit = os.environ.get("POLICYENGINE_UK_COMMIT")
    return f"{version}+git.{commit[:8]}" if commit else version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate results/lvt_results.json from one baseline run."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset path or hf:// URL (default: $UK_LVT_DATASET or the pin).",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--uk-data-root",
        type=Path,
        default=None,
        help="Local policyengine-uk-data checkout holding the ONS land targets.",
    )
    return parser


def write_results(results: dict, output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    return output


def main(argv: list[str] | None = None) -> int:
    global DATASET

    args = build_parser().parse_args(argv)
    if args.dataset:
        DATASET = args.dataset
    results = build_results(uk_data_root=args.uk_data_root)
    write_results(results, args.output)
    print(f"Wrote {args.output} (policyengine-uk {results['model_version']})")
    print(json.dumps(results["council_tax_replacement"], indent=2))
    print(json.dumps(results["poverty_gini"]["scenarios"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
