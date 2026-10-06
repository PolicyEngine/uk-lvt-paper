"""Raw land share per English authority (prototype).

    plot land per new dwelling = V / (f * d)
    raw lambda                 = plot land per new dwelling / stock price

* ``V``: MHCLG land value estimate 2023, GBP per hectare, for a serviced 1 ha
  residential site (central case: "Residential (adj for floor)" sheet, medium
  density scenario, median new-build price percentile).
* ``d``: MHCLG's published dwellings per hectare for the same density
  scenario.
* ``f``: density factor. 1 in the central case. The guidance does not say
  whether ``d`` is per gross or per net hectare; under the net reading a
  gross hectare holds 0.8 d homes outside London (80% net developable area),
  i.e. f = 0.8 and lambda is 25% higher. The sensitivity runs f = 0.8
  (lambda x 1.25) and f = 4/3 (lambda x 0.75) for every authority.
* ``stock price``: prototype proxy, the UK HPI average price for the
  authority in October 2023 (MHCLG's valuation date is 1 October 2023). The
  Price Paid route is not built yet; see ``docs/land_methods.md``.

Lambda is reported on [0, 1]: a ratio above 1 is capped at 1 and the variant
is named in ``capped_variants``. Values are raw: they are not yet scaled to
the ONS anchor.

``python -m uk_lvt.land.lambda_raw`` rebuilds ``results/land/`` and the
generated tables in ``docs/land_methods.md`` from the committed tables in
``uk_lvt/land/data/``; ``--check`` fails if any of them would change.
"""

from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import DERIVED_DIR, METHODS_NOTE_PATH, RESULTS_DIR
from .derive import CTSOP_FILE, HPI_FILE, MHCLG_FILE
from .geography import (
    ENGLAND_CODE,
    IMPUTATION_DONORS,
    MHCLG_TO_2023,
    N_ENGLISH_AUTHORITIES,
    check_universe,
    crosswalk_mhclg,
)

HPI_DATE = "2023-10-01"
NET_DEVELOPABLE_SHARE = 0.8


@dataclass(frozen=True)
class Variant:
    name: str
    sheet: str
    density_scenario: str
    price_percentile: int
    density_factor: float
    description: str


VARIANTS: tuple[Variant, ...] = (
    Variant("central", "floor_adjusted", "medium", 50, 1.0,
            "Floor-adjusted sheet, medium (central) density, median new-build price percentile"),
    Variant("density_minus25", "floor_adjusted", "medium", 50, 4.0 / 3.0,
            "As central with 4/3 x published density (lambda x 0.75)"),
    Variant("density_plus25", "floor_adjusted", "medium", 50, NET_DEVELOPABLE_SHARE,
            "As central with 0.8 x published density (lambda x 1.25; the net-hectare reading outside London)"),
    Variant("price_p25", "floor_adjusted", "medium", 25, 1.0,
            "As central at MHCLG's 25th new-build price percentile"),
    Variant("price_p75", "floor_adjusted", "medium", 75, 1.0,
            "As central at MHCLG's 75th new-build price percentile"),
    Variant("unadjusted", "unadjusted", "medium", 50, 1.0,
            "As central from the sheet without MHCLG's floor value; missing where MHCLG reports no value"),
)
VARIANT_NAMES = tuple(v.name for v in VARIANTS)
WEIGHTINGS = ("dwellings", "value", "unweighted")

LA_FILE = "la_lambda_raw.csv"
SUMMARY_FILE = "la_lambda_raw_summary.csv"


# --------------------------------------------------------------------------
# Pure arithmetic (property-tested)
# --------------------------------------------------------------------------

def plot_land_per_dwelling(land_value_per_ha, dwellings_per_ha, density_factor=1.0):
    """V / (f * d). NaN in V passes through; d and f must be positive."""
    v = np.asarray(land_value_per_ha, dtype=float)
    d = np.asarray(dwellings_per_ha, dtype=float)
    f = np.asarray(density_factor, dtype=float)
    if np.any(~(d > 0)):
        raise ValueError("dwellings per hectare must be positive")
    if np.any(~(f > 0)):
        raise ValueError("density factor must be positive")
    if np.any(v[~np.isnan(v)] < 0):
        raise ValueError("land value per hectare must be non-negative")
    return v / (f * d)


def raw_ratio(plot_land, stock_price):
    """Plot land per dwelling over stock price (unbounded)."""
    p = np.asarray(stock_price, dtype=float)
    if np.any(~(p > 0)):
        raise ValueError("stock price must be positive")
    return np.asarray(plot_land, dtype=float) / p


def cap_unit_interval(ratio):
    """Return ``(lambda, capped)``: ratio capped at 1, and where it was capped.

    NaN stays NaN (not capped). A negative ratio is invalid input.
    """
    r = np.asarray(ratio, dtype=float)
    if np.any(r[~np.isnan(r)] < 0):
        raise ValueError("land share ratio must be non-negative")
    capped = np.where(np.isnan(r), False, r > 1.0)
    lam = np.where(capped, 1.0, r)
    return lam, capped


def lambda_from_inputs(land_value_per_ha, dwellings_per_ha, stock_price, density_factor=1.0):
    """Raw lambda in [0, 1] (or NaN) and the capped mask."""
    ratio = raw_ratio(
        plot_land_per_dwelling(land_value_per_ha, dwellings_per_ha, density_factor),
        stock_price,
    )
    return cap_unit_interval(ratio)


def convex_weights(weights) -> np.ndarray:
    """Normalise non-negative weights with a positive sum to sum to 1."""
    w = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    total = w.sum()
    if not total > 0:
        raise ValueError("weights must have a positive sum")
    return w / total


def weighted_mean(values, weights) -> float:
    """Weighted mean over the non-NaN values: a convex combination of them."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != w.shape:
        raise ValueError("values and weights differ in shape")
    keep = ~np.isnan(x)
    if not keep.any():
        return float("nan")
    alpha = convex_weights(w[keep])
    return float(np.dot(alpha, x[keep]))


# --------------------------------------------------------------------------
# Authority table
# --------------------------------------------------------------------------

def load_inputs(data_dir: Path = DERIVED_DIR):
    mhclg = pd.read_csv(data_dir / MHCLG_FILE, dtype={"la_code_mhclg": str})
    hpi = pd.read_csv(data_dir / HPI_FILE, dtype={"la_code": str, "date": str})
    ctsop = pd.read_csv(data_dir / CTSOP_FILE, dtype={"la_code": str, "billing_authority_code": str})
    return mhclg, hpi, ctsop


def _mhclg_wide(mhclg: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    sel = mhclg[
        (mhclg["sheet"] == variant.sheet)
        & (mhclg["density_scenario"] == variant.density_scenario)
        & (mhclg["price_percentile"] == variant.price_percentile)
    ]
    if sel["la_code"].duplicated().any():
        raise ValueError(f"{variant.name}: duplicate MHCLG rows")
    return sel.set_index("la_code")[["la_code_mhclg", "dwellings_per_ha", "land_value_gbp_per_ha"]]


def _int_if_integral(col: pd.Series) -> pd.Series:
    """Nullable integers when every value is whole (as published), else floats."""
    vals = col.astype(float)
    finite = vals.dropna()
    if (finite == np.floor(finite)).all():
        return vals.astype("Int64")
    return vals


def build_la_table(mhclg: pd.DataFrame, hpi: pd.DataFrame, ctsop: pd.DataFrame,
                   hpi_date: str = HPI_DATE,
                   expected_n: int | None = N_ENGLISH_AUTHORITIES) -> pd.DataFrame:
    """One row per authority in ``ctsop`` (296 for England), with raw lambda and sensitivities."""
    universe = ctsop[["la_code", "la_name", "region_code", "region_name", "all_properties"]].copy()
    check_universe(universe["la_code"], expected_n)
    universe = universe.set_index("la_code")

    mhclg = crosswalk_mhclg(mhclg)
    unknown = sorted(set(mhclg["la_code"]) - set(universe.index))
    if unknown:
        raise ValueError(f"MHCLG authorities outside the universe: {unknown}")

    prices = hpi[hpi["date"] == hpi_date]
    if prices["la_code"].duplicated().any():
        raise ValueError("duplicate HPI rows")
    unknown = sorted(set(prices["la_code"]) - set(universe.index))
    if unknown:
        raise ValueError(f"HPI authorities outside the universe: {unknown}")
    price = prices.set_index("la_code")["average_price"].astype(float)

    central = _mhclg_wide(mhclg, VARIANTS[0])
    out = pd.DataFrame(index=universe.index)
    out["la_name"] = universe["la_name"]
    out["region_code"] = universe["region_code"]
    out["region_name"] = universe["region_name"]
    out["dwellings"] = universe["all_properties"].astype(int)
    out["mhclg_code"] = central["la_code_mhclg"].reindex(out.index)
    out["stock_price"] = price.reindex(out.index)
    out["density_central_dph"] = central["dwellings_per_ha"].reindex(out.index)
    out["land_value_central_gbp_per_ha"] = central["land_value_gbp_per_ha"].reindex(out.index)

    computed = out["stock_price"].notna() & out["land_value_central_gbp_per_ha"].notna()
    rows = computed.to_numpy()
    plot = np.full(len(out), np.nan)
    plot[rows] = plot_land_per_dwelling(
        out["land_value_central_gbp_per_ha"].to_numpy(dtype=float)[rows],
        out["density_central_dph"].to_numpy(dtype=float)[rows],
    )
    out["plot_land_per_dwelling_gbp"] = plot

    capped_lists = {code: [] for code in out.index}
    for variant in VARIANTS:
        wide = _mhclg_wide(mhclg, variant).reindex(out.index)
        lam = np.full(len(out), np.nan)
        capped = np.zeros(len(out), dtype=bool)
        lam[rows], capped[rows] = lambda_from_inputs(
            wide["land_value_gbp_per_ha"].to_numpy(dtype=float)[rows],
            wide["dwellings_per_ha"].to_numpy(dtype=float)[rows],
            out["stock_price"].to_numpy(dtype=float)[rows],
            variant.density_factor,
        )
        out[f"lambda_raw_{variant.name}"] = lam
        for code in out.index[capped]:
            capped_lists[code].append(variant.name)

    # Donor imputation (Isles of Scilly takes Cornwall's inputs and results).
    out["method_flag"] = np.where(
        out["mhclg_code"].isin(list(MHCLG_TO_2023)), "computed_code_crosswalk", "computed"
    )
    copy_cols = ["stock_price", "density_central_dph", "land_value_central_gbp_per_ha",
                 "plot_land_per_dwelling_gbp"] + [f"lambda_raw_{v}" for v in VARIANT_NAMES]
    for code, donor in IMPUTATION_DONORS.items():
        if code not in out.index or computed.get(code, False):
            continue
        if not computed.get(donor, False):
            raise ValueError(f"donor {donor} for {code} has no computed values")
        out.loc[code, copy_cols] = out.loc[donor, copy_cols]
        capped_lists[code] = list(capped_lists[donor])
        out.loc[code, "method_flag"] = f"imputed_from_{donor}"
    missing = out.index[out["lambda_raw_central"].isna()].tolist()
    if missing:
        raise ValueError(f"no central lambda for {missing}")
    out["capped_variants"] = [";".join(capped_lists[c]) for c in out.index]

    out = out.reset_index().rename(columns={"index": "la_code"})
    out = out.rename(columns={"lambda_raw_central": "lambda_raw"})
    out["plot_land_per_dwelling_gbp"] = out["plot_land_per_dwelling_gbp"].round(2)
    for col in [c for c in out.columns if c.startswith("lambda_raw")]:
        out[col] = out[col].round(6)
    out["density_central_dph"] = _int_if_integral(out["density_central_dph"])
    out["land_value_central_gbp_per_ha"] = _int_if_integral(out["land_value_central_gbp_per_ha"])
    out["stock_price"] = out["stock_price"].round(0).astype("Int64")
    order = ["la_code", "la_name", "region_code", "region_name", "mhclg_code", "dwellings",
             "stock_price", "density_central_dph", "land_value_central_gbp_per_ha",
             "plot_land_per_dwelling_gbp", "lambda_raw"]
    order += [f"lambda_raw_{v}" for v in VARIANT_NAMES[1:]]
    order += ["capped_variants", "method_flag"]
    return out[order].sort_values(["region_code", "la_code"]).reset_index(drop=True)


def variant_column(name: str) -> str:
    return "lambda_raw" if name == "central" else f"lambda_raw_{name}"


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _weights(frame: pd.DataFrame, weighting: str) -> np.ndarray:
    if weighting == "dwellings":
        return frame["dwellings"].to_numpy(dtype=float)
    if weighting == "value":
        return frame["dwellings"].to_numpy(dtype=float) * frame["stock_price"].to_numpy(dtype=float)
    if weighting == "unweighted":
        return np.ones(len(frame))
    raise ValueError(weighting)


def summarise(la: pd.DataFrame) -> pd.DataFrame:
    """England and regional aggregates of every variant under each weighting."""
    groups = [(ENGLAND_CODE, "England", la)]
    for (code, name), g in la.groupby(["region_code", "region_name"], sort=True):
        groups.append((code, name, g))
    records = []
    for code, name, g in groups:
        for variant in VARIANT_NAMES:
            col = variant_column(variant)
            vals = g[col].to_numpy(dtype=float)
            n_capped = int(g["capped_variants"].fillna("").str.split(";").map(lambda xs: variant in xs).sum())
            for weighting in WEIGHTINGS:
                records.append({
                    "geography_code": code,
                    "geography_name": name,
                    "variant": variant,
                    "weighting": weighting,
                    "lambda_raw": round(weighted_mean(vals, _weights(g, weighting)), 6),
                    "n_authorities": int((~np.isnan(vals)).sum()),
                    "n_missing": int(np.isnan(vals).sum()),
                    "n_capped": n_capped,
                    "dwellings": int(g["dwellings"].sum()),
                })
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------
# Methods-note tables (generated, so every number in the note traces here)
# --------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{x:.3f}"


def render_regional_table(la: pd.DataFrame, summary: pd.DataFrame) -> str:
    s = summary[summary["variant"] == "central"].pivot_table(
        index=["geography_code", "geography_name"], columns="weighting", values="lambda_raw"
    ).reset_index()
    s["order"] = (s["geography_code"] != ENGLAND_CODE).astype(int)
    s = s.sort_values(["order", "geography_code"])
    lines = [
        "| Area | Authorities | Dwelling-weighted | Value-weighted | Unweighted mean | Lowest | Highest |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in s.itertuples(index=False):
        g = la if row.geography_code == ENGLAND_CODE else la[la["region_code"] == row.geography_code]
        lines.append(
            f"| {row.geography_name} | {len(g)} | {_fmt(row.dwellings)} | {_fmt(row.value)} | "
            f"{_fmt(row.unweighted)} | {_fmt(g['lambda_raw'].min())} | {_fmt(g['lambda_raw'].max())} |"
        )
    return "\n".join(lines)


def render_sensitivity_table(summary: pd.DataFrame) -> str:
    eng = summary[summary["geography_code"] == ENGLAND_CODE]
    lines = [
        "| Variant | Definition | England, dwelling-weighted | England, value-weighted | Authorities with a value | Capped at 1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        rows = eng[eng["variant"] == v.name].set_index("weighting")
        lines.append(
            f"| `{v.name}` | {v.description} | {_fmt(rows.loc['dwellings', 'lambda_raw'])} | "
            f"{_fmt(rows.loc['value', 'lambda_raw'])} | {int(rows.loc['dwellings', 'n_authorities'])} | "
            f"{int(rows.loc['dwellings', 'n_capped'])} |"
        )
    return "\n".join(lines)


def render_extremes_table(la: pd.DataFrame, n: int = 5) -> str:
    ranked = la.sort_values(["lambda_raw", "la_code"])
    lines = ["| Rank | Lowest raw lambda | Value | Highest raw lambda | Value |", "|---:|---|---:|---|---:|"]
    low = ranked.head(n).itertuples(index=False)
    high = ranked.iloc[::-1].head(n).itertuples(index=False)
    for i, (lo, hi) in enumerate(zip(low, high), start=1):
        lines.append(f"| {i} | {lo.la_name} | {_fmt(lo.lambda_raw)} | {hi.la_name} | {_fmt(hi.lambda_raw)} |")
    return "\n".join(lines)


GENERATED_BLOCKS = {
    "regional_table": render_regional_table,
    "sensitivity_table": lambda la, summary: render_sensitivity_table(summary),
    "extremes_table": lambda la, summary: render_extremes_table(la),
}


def replace_generated_blocks(text: str, la: pd.DataFrame, summary: pd.DataFrame) -> str:
    for key, render in GENERATED_BLOCKS.items():
        start = f"<!-- BEGIN GENERATED: {key} -->"
        end = f"<!-- END GENERATED: {key} -->"
        i, j = text.find(start), text.find(end)
        if i < 0 or j < 0 or j < i:
            raise ValueError(f"methods note lacks the {key} markers")
        text = text[: i + len(start)] + "\n" + render(la, summary) + "\n" + text[j:]
    return text


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _csv_text(frame: pd.DataFrame) -> str:
    buf = io.StringIO()
    frame.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def build_outputs(data_dir: Path = DERIVED_DIR) -> dict[str, str]:
    """Return ``{name: text}`` for both result CSVs (nothing written)."""
    la = build_la_table(*load_inputs(data_dir))
    summary = summarise(la)
    return {LA_FILE: _csv_text(la), SUMMARY_FILE: _csv_text(summary)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DERIVED_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--methods-note", type=Path, default=METHODS_NOTE_PATH)
    ap.add_argument("--check", action="store_true", help="fail if any output would change")
    args = ap.parse_args(argv)

    la = build_la_table(*load_inputs(args.data_dir))
    summary = summarise(la)
    targets = {
        args.results_dir / LA_FILE: _csv_text(la),
        args.results_dir / SUMMARY_FILE: _csv_text(summary),
    }
    note = args.methods_note.read_text()
    targets[args.methods_note] = replace_generated_blocks(note, la, summary)

    stale = [p for p, text in targets.items() if not p.exists() or p.read_text() != text]
    if args.check:
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    for p, text in targets.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    eng = summary[(summary["geography_code"] == ENGLAND_CODE) & (summary["variant"] == "central")]
    print(f"{len(la)} authorities; England central raw lambda:")
    for row in eng.itertuples(index=False):
        print(f"  {row.weighting:<11} {row.lambda_raw:.4f}")
    print(f"wrote {', '.join(str(p) for p in targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
