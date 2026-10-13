"""Stock-reweighted transaction residuals; rebuild outputs from small cells.

Raw derivation: ``python -m uk_lvt.land.residual_derive``.
Offline results: ``python -m uk_lvt.land.residual [--check]``.
See docs/land_residual_method.md for proxy bands and coverage limitations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import DERIVED_DIR, REPO_ROOT
from .fetch import HashMismatch, sha256_file
from .geography import check_universe
from .lambda_raw import weighted_mean
from .manifest import load_manifest

VALUATION_DATE = "2026-04-01"
RESULT_DIR = REPO_ROOT / "results/land"
RESIDUAL_MANIFEST = DERIVED_DIR / "residual_manifest.json"


def residual_share(price: float, area: float, cost_per_m2: float) -> float:
    """Replacement-cost residual, with no depreciation or national scaling."""
    if not np.isfinite(price) or price <= 0:
        raise ValueError("price must be finite and positive")
    if not np.isfinite(area) or area < 0 or not np.isfinite(cost_per_m2) or cost_per_m2 < 0:
        raise ValueError("area and cost must be finite and nonnegative")
    if area == 0 or cost_per_m2 == 0:
        return 1.0
    # Divide first to avoid an overflowing area*cost becoming NaN at zero.
    return float(max(0.0, 1.0 - (area / price) * cost_per_m2))


def verify_inputs(cache_dir: Path | None = None) -> dict:
    """Always verify derived inputs and source pins; verify raw files if present."""
    meta = json.loads(RESIDUAL_MANIFEST.read_text())
    sources, top = load_manifest()
    for sid, digest in meta["sources"].items():
        if sources[sid].sha256 != digest:
            raise HashMismatch(f"{sid}: residual provenance differs from sources.yaml")
        raw = (cache_dir or REPO_ROOT / top["default_cache_dir"]) / sources[sid].filename
        if raw.exists() and sha256_file(raw) != digest:
            raise HashMismatch(f"{sid}: cached input hash differs")
    for name, info in meta["tables"].items():
        if sha256_file(DERIVED_DIR / name) != info["sha256"]:
            raise HashMismatch(f"{name}: derived input hash differs")
    for name, digest in meta["prototype_tables"].items():
        if sha256_file(REPO_ROOT / name) != digest:
            raise HashMismatch(f"{name}: prototype input hash differs")
    return meta


def authority_table(cells: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Dwelling-weighted mean of individual ratios; retain ratio of totals too."""
    rows = []
    for u in universe.itertuples(index=False):
        c = cells[cells.la_code.eq(u.la_code)]
        if c.empty or not np.isclose(c.stock.sum(), u.all_properties):
            raise ValueError(f"{u.la_code}: cell stock does not cover authority")
        row = dict(la_code=u.la_code, la_name=u.la_name, region_name=u.region_name,
                   dwellings=u.all_properties, valuation_date=VALUATION_DATE)
        for source, target in [("mean_share", "lambda_residual"),
                               ("mean_share_cost_low", "lambda_cost_minus25"),
                               ("mean_share_cost_high", "lambda_cost_plus25"),
                               ("mean_price", "stock_mean_price_gbp"),
                               ("mean_land", "stock_mean_land_gbp")]:
            row[target] = weighted_mean(c[source], c.stock)
        row["lambda_value_weighted"] = row["stock_mean_land_gbp"] / row["stock_mean_price_gbp"]
        row["implied_land_gbp"] = row["stock_mean_land_gbp"] * u.all_properties
        row["implied_property_gbp"] = row["stock_mean_price_gbp"] * u.all_properties
        row["latest_sales"] = int(c.sales.sum())
        row["sales_clipped"] = int(c.sales_clipped.sum())
        row["fallback_stock_fraction"] = c.loc[c.fallback.ne("observed"), "stock"].sum() / c.stock.sum()
        # Sensitivity: type-only weights, without the imputed price-rank bands.
        type_means, type_weights = [], []
        for _, t in c.groupby("property_type", sort=True):
            if t.stock.sum() == 0:
                continue
            observed = t[t.sales.gt(0)]
            type_weights.append(t.stock.sum())
            type_means.append(weighted_mean(observed.mean_share, observed.sales)
                              if not observed.empty else weighted_mean(t.mean_share, t.stock))
        row["lambda_type_only"] = weighted_mean(type_means, type_weights)
        row["method_flag"] = "rank_proxy_bands"
        if c.loc[c.stock.gt(0), "fallback"].ne("observed").any():
            row["method_flag"] += ";sparse_cell_fallback"
        if u.la_code == "E06000053":
            row["method_flag"] += ";scilly_cornwall_hpi"
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("la_code").reset_index(drop=True)
    check_universe(out.la_code)
    if set(out.la_code) != set(universe.la_code):
        raise ValueError("authority codes differ from prototype universe")
    return out


def comparisons(authorities: pd.DataFrame, tpa: pd.DataFrame, mhclg: pd.DataFrame):
    table = authorities.merge(tpa[["la_code", "tpa_land_value_share"]], on="la_code", validate="one_to_one")
    table = table.merge(mhclg[["la_code", "lambda_raw"]], on="la_code", validate="one_to_one")
    check_universe(table.la_code)
    correlations = {}
    disagreements = []
    for reference, label in [("tpa_land_value_share", "tpa"), ("lambda_raw", "mhclg")]:
        # pandas Pearson correlation of average ranks is Spearman, including ties.
        correlations[label] = float(table.lambda_residual.rank().corr(table[reference].rank()))
        correlations[label + "_value_weighted"] = float(table.lambda_value_weighted.rank().corr(table[reference].rank()))
        top = table.assign(reference=label, reference_share=table[reference],
                           difference=table.lambda_residual - table[reference])
        top = top.assign(absolute_difference=top.difference.abs()).sort_values(
            ["absolute_difference", "la_code"], ascending=[False, True]).head(20)
        disagreements.append(top[["reference", "la_code", "la_name", "region_name", "lambda_residual",
                                  "reference_share", "difference", "absolute_difference"]])
    summaries = []
    for name, group in [("England", table), *table.groupby("region_name", sort=True)]:
        summaries.append(dict(area=name, authorities=len(group), dwellings=int(group.dwellings.sum()),
                              residual_dwelling_weighted=weighted_mean(group.lambda_residual, group.dwellings),
                              residual_value_weighted=group.implied_land_gbp.sum() / group.implied_property_gbp.sum(),
                              tpa_dwelling_weighted=weighted_mean(group.tpa_land_value_share, group.dwellings),
                              mhclg_dwelling_weighted=weighted_mean(group.lambda_raw, group.dwellings),
                              implied_land_gbp=group.implied_land_gbp.sum(),
                              implied_property_gbp=group.implied_property_gbp.sum()))
    return table, pd.concat(disagreements, ignore_index=True), pd.DataFrame(summaries), correlations


def output_payloads() -> dict[Path, str]:
    meta = verify_inputs()
    universe = pd.read_csv(DERIVED_DIR / "voa_ctsop1_2026_england_la.csv")
    cells = pd.read_csv(DERIVED_DIR / "residual_cells.csv", keep_default_na=False)
    authorities = authority_table(cells, universe)
    table, disagreements, regions, correlations = comparisons(
        authorities, pd.read_csv(DERIVED_DIR / "tpa_residual_reference.csv"),
        pd.read_csv(RESULT_DIR / "la_lambda_raw.csv"))
    england = regions.iloc[0]
    summary = dict(valuation_date=VALUATION_DATE, spearman={k: round(v, 9) for k, v in correlations.items()},
                   latest_sales=int(table.latest_sales.sum()), sales_clipped=int(table.sales_clipped.sum()),
                   implied_england_land_gbp=round(float(england.implied_land_gbp)),
                   implied_england_property_gbp=round(float(england.implied_property_gbp)),
                   dwelling_weighted_share=round(float(england.residual_dwelling_weighted), 9),
                   value_weighted_share=round(float(england.residual_value_weighted), 9),
                   ons=meta["ons"], coverage=meta["coverage"],
                   england_2026_to_uk_2021_land_ratio=round(float(england.implied_land_gbp) / meta["ons"]["historical_uk_dwelling_land_gbp"], 9),
                   comparison_warning="England April 2026 versus UK end-2021: geography and date differ; latest dwelling-land subclass unavailable")
    return {
        RESULT_DIR / "la_lambda_residual.csv": csv_payload(table),
        RESULT_DIR / "land_residual_disagreements.csv": csv_payload(disagreements),
        RESULT_DIR / "land_residual_regions.csv": csv_payload(regions),
        RESULT_DIR / "land_residual_summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
    }


def csv_payload(frame: pd.DataFrame) -> str:
    """Report totals to pounds and dwelling means to pence, avoiding false precision.

    Aggregations use the unrounded cell means. Rounding only serialized outputs
    also prevents insignificant BLAS/platform differences changing golden files.
    """
    out = frame.copy()
    for column in out.columns:
        if column.startswith("implied_") and column.endswith("_gbp"):
            out[column] = out[column].round(0)
        elif column.endswith("_gbp"):
            out[column] = out[column].round(2)
    return out.to_csv(index=False, float_format="%.9f", lineterminator="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    changed = []
    for path, content in output_payloads().items():
        if not path.exists() or path.read_text() != content:
            changed.append(str(path.relative_to(REPO_ROOT)))
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
    print(("would change: " if args.check else "wrote: ") + ", ".join(changed) if changed else "residual outputs match")
    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
