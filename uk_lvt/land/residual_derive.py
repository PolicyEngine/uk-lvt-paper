"""Derive small, address-free residual cells from pinned yearly Price Paid files.

Requires the land extra; DuckDB uses two threads and a 512 MB memory limit.
Nothing is downloaded implicitly: use uk_lvt.land.fetch --only with the source
IDs in data/residual_manifest.json. Raw files and scratch tables stay in cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import DERIVED_DIR, REPO_ROOT
from .fetch import cached_path, default_cache_dir, sha256_file
from .manifest import load_manifest
from .residual import RESIDUAL_MANIFEST, VALUATION_DATE
from .residual_inputs import normalize_district_name, parse_ctsop3_stock, ppd_district_crosswalk
from .residual_reference import parse_floor_areas, parse_ons_dwelling_land, parse_rebuild_constants, parse_tpa_cells

YEARS = tuple(range(2018, 2027))
SOURCE_IDS = [*(f"hmlr_price_paid_{year}" for year in YEARS),
              "ukhpi_average_prices_2026_07", "voa_ctsop_2026_summary", "ehs_2018_floor_areas",
              "tpa_rebuild_config", "tpa_lvt_model_cells", "tpa_property_value_method",
              "tpa_cell_method", "tpa_mit_licence", "ons_preliminary_nbs_2026",
              "ons_nbs_timeseries_2026", "ons_dwelling_land_2021"]
STAT_COLUMNS = ["mean_price", "mean_land", "mean_share", "mean_share_cost_low", "mean_share_cost_high"]
PPD_COLUMNS = [
    "transaction_id", "price", "sale_date", "postcode", "property_type", "new_build",
    "tenure", "paon", "saon", "street", "locality", "town", "district", "county",
    "category", "record_status",
]


def read_price_paid(con, csv_paths: list[str]) -> None:
    """Register the 16-column yearly PPD CSVs with their published field order."""
    con.read_csv(csv_paths, header=False, columns={name: "VARCHAR" for name in PPD_COLUMNS},
                 na_values="", quotechar='"').create_view("raw")


def create_latest_sales(con) -> None:
    """Select each address's latest eligible deed in the bounded input window.

    The caller registers ``raw`` and the exact district-name ``crosswalk``.
    Identical sale dates use the transaction ID as a deterministic tie-break.
    Annual snapshots contain current records; category A is a standard-market
    proxy, rather than independent proof that every sale was at arm's length.
    """
    con.execute("""create table latest as
        select transaction_id, price::double AS price, substr(sale_date,1,7)||'-01' AS "month",
               sale_date, property_type, tenure, new_build, x.la_code,
               sha256(coalesce(postcode,'')||chr(31)||coalesce(paon,'')||chr(31)||coalesce(saon,'')||chr(31)||coalesce(street,'')) AS property_key
        from raw r join crosswalk x using(district)
        where x.la_code is not null and category='A' and record_status in ('A','C')
          and property_type in ('D','S','T','F') and price::double>0
          and postcode is not null and paon is not null
          and substr(sale_date,1,10) between '2018-01-01' and '2026-04-30'
        qualify row_number() over(partition by postcode,paon,saon,street
                                  order by sale_date desc,transaction_id desc)=1""")


def create_priced_sales(con) -> None:
    """Apply authority/month HPI uprating and authority/type replacement cost."""
    con.execute("""create table priced as select l.*, price*target_hpi/hpi AS "value",
                   rebuild.rebuild_cost AS "cost" from latest l
                   join hpi using(la_code,"month") join target using(la_code)
                   join rebuild using(la_code,property_type)""")


def create_ranked_sales(con) -> None:
    """Rank current prices within each authority using deterministic midpoints."""
    con.execute("""create table ranked as select *,
              (row_number() over(partition by la_code order by "value",property_key)-0.5)
              / count(*) over(partition by la_code) AS "quantile" from priced""")


def observed_cell_statistics(con) -> pd.DataFrame:
    """Aggregate sale ratios and clipped residuals into stock proxy-band cells."""
    return con.execute("""select la_code,property_type,band,count(*) AS sales,
             count(*) filter(where "value"<cost) AS sales_clipped,
             avg("value") AS mean_price,avg(greatest(0,"value"-cost)) AS mean_land,
             avg(greatest(0,1-cost/"value")) AS mean_share,
             avg(greatest(0,1-0.75*cost/"value")) AS mean_share_cost_low,
             avg(greatest(0,1-1.25*cost/"value")) AS mean_share_cost_high
             from ranked join bands using(la_code)
             where "quantile">="lower" and "quantile"<"upper"
             group by la_code,property_type,band order by la_code,property_type,band""").fetchdf()


def assign_proxy_bands(prices, band_counts) -> np.ndarray:
    """Assign overall LA price ranks to stock band quantiles, not observed bands."""
    prices = np.asarray(prices, dtype=float)
    counts = np.asarray(band_counts, dtype=float)
    if len(counts) != 8 or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("eight nonnegative band counts with positive sum required")
    if not len(prices):
        return np.array([], dtype=str)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be positive and finite")
    # Stable tie handling is deterministic, although identical-price homes can
    # straddle a boundary. This is a calibration assumption, not VOA matching.
    order = np.argsort(prices, kind="stable")
    positions = (np.arange(len(prices)) + 0.5) / len(prices)
    bands = np.minimum(np.searchsorted(np.cumsum(counts) / counts.sum(), positions, side="right"), 7)
    result = np.empty(len(prices), dtype="U1")
    result[order] = np.array(list("ABCDEFGH"))[bands]
    return result


def fill_sparse_cells(stock: pd.DataFrame, observed: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Fill unobserved stock cells using LA type, then region type sale means."""
    out = stock.merge(observed, on=["la_code", "property_type", "band"], how="left", validate="one_to_one")
    regions = universe.set_index("la_code").region_name.to_dict()
    obs = observed.assign(region_name=observed.la_code.map(regions))

    def pooled(frame):
        return {col: np.average(frame[col], weights=frame.sales) for col in STAT_COLUMNS}

    local = {key: pooled(group) for key, group in obs.groupby(["la_code", "property_type"])}
    regional = {key: pooled(group) for key, group in obs.groupby(["region_name", "property_type"])}
    records = []
    for row in out.to_dict("records"):
        if pd.notna(row["sales"]):
            row["fallback"] = "observed"
        else:
            key = (row["la_code"], row["property_type"])
            if key in local:
                donor = local[key]
                row["fallback"] = "la_type"
            else:
                donor = regional[(regions[row["la_code"]], row["property_type"])]
                row["fallback"] = "region_type"
            row.update(donor)
            row["sales"] = row["sales_clipped"] = 0
        records.append(row)
    result = pd.DataFrame(records)
    result[["sales", "sales_clipped"]] = result[["sales", "sales_clipped"]].astype(int)
    return result.sort_values(["la_code", "property_type", "band"]).reset_index(drop=True)


def derive(cache: Path):
    import duckdb

    print("Verifying pinned inputs", flush=True)
    paths = {sid: cached_path(sid, cache) for sid in SOURCE_IDS}
    universe = pd.read_csv(DERIVED_DIR / "voa_ctsop1_2026_england_la.csv")
    stock = parse_ctsop3_stock(paths["voa_ctsop_2026_summary"], universe)
    stock_diagnostics = stock.attrs.copy()
    areas = parse_floor_areas(paths["ehs_2018_floor_areas"])
    costs, region_factors = parse_rebuild_constants(paths["tpa_rebuild_config"])
    costs_table = universe[["la_code", "region_name", "region_code"]].merge(areas, on="region_name", validate="many_to_many")
    costs_table["cost_per_m2_gbp"] = costs_table.property_type.map(costs) * costs_table.region_code.map(region_factors)
    costs_table["rebuild_cost"] = costs_table.floor_area_m2 * costs_table.cost_per_m2_gbp
    if costs_table.rebuild_cost.isna().any():
        raise ValueError("unmapped floor area or rebuild cost")

    # Actual district names are few, so discover and normalise in Python. The
    # property rows themselves never leave the memory-limited database.
    scratch = cache / "residual_scratch.duckdb"
    scratch.unlink(missing_ok=True)
    con = duckdb.connect(str(scratch), config={"threads": 2, "memory_limit": "512MB"})
    try:
        csvs = [str(paths[f"hmlr_price_paid_{year}"]) for year in YEARS]
        read_price_paid(con, csvs)
        print("Reading yearly Price Paid files and mapping districts", flush=True)
        district_map = ppd_district_crosswalk(universe)
        districts = [x[0] for x in con.execute("select distinct district from raw").fetchall()]
        crosswalk = pd.DataFrame([dict(district=d, la_code=district_map.get(normalize_district_name(d))) for d in districts])
        con.register("crosswalk", crosswalk)
        unmatched = con.execute("select r.district,count(*) AS sales from raw r left join crosswalk x using(district) where x.la_code is null group by r.district order by r.district").fetchdf().to_dict("records")
        raw_rows = con.execute("select count(*) from raw").fetchone()[0]
        create_latest_sales(con)
        print("Latest-sale records selected; applying HPI and rebuild costs", flush=True)
        hpi = pd.read_csv(paths["ukhpi_average_prices_2026_07"])
        hpi = hpi.rename(columns={"Date": "month", "Area_Code": "la_code", "Average_Price": "hpi"})
        hpi = hpi[hpi.la_code.isin(universe.la_code)][["la_code", "month", "hpi"]]
        # Same prototype donor for Scilly, retaining Scilly's transactions.
        hpi = pd.concat([hpi, hpi[hpi.la_code.eq("E06000052")].assign(la_code="E06000053")], ignore_index=True)
        target = hpi[hpi.month.eq(VALUATION_DATE)][["la_code", "hpi"]].rename(columns={"hpi": "target_hpi"})
        if target.la_code.duplicated().any() or len(target) != 296 or hpi.duplicated(["la_code", "month"]).any():
            raise ValueError("HPI lacks unique authority/month coverage")
        con.register("hpi", hpi)
        con.register("target", target)
        con.register("rebuild", costs_table)
        create_priced_sales(con)
        print("Assigning proxy bands and deriving residual cells", flush=True)
        latest_rows = con.execute("select count(*) from latest").fetchone()[0]
        priced_rows = con.execute("select count(*) from priced").fetchone()[0]
        if priced_rows != latest_rows:
            raise ValueError(f"HPI/rebuild coverage lost or duplicated {latest_rows-priced_rows} sales")
        # Midpoint ranks reproduce the scalar assign_proxy_bands definition.
        create_ranked_sales(con)
        band_weights = stock.groupby(["la_code", "band"], as_index=False).stock.sum()
        band_weights["upper"] = band_weights.groupby("la_code").stock.cumsum() / band_weights.groupby("la_code").stock.transform("sum")
        band_weights["lower"] = band_weights.groupby("la_code")["upper"].shift().fillna(0)
        con.register("bands", band_weights)
        observed = observed_cell_statistics(con)
        if int(observed.sales.sum()) != priced_rows:
            raise ValueError("rank bands lost or duplicated sales")
        coverage = dict(raw_rows=raw_rows, latest_sales=latest_rows, sales_clipped=int(observed.sales_clipped.sum()),
                        transaction_years=list(YEARS), latest_deed_date="2026-04-30",
                        unmatched_districts=unmatched, stock_conversion=stock_diagnostics,
                        leasehold_sales=con.execute("select count(*) from priced where tenure='L'").fetchone()[0],
                        new_build_sales=con.execute("select count(*) from priced where new_build='Y'").fetchone()[0])
        cells = fill_sparse_cells(stock, observed, universe)
    finally:
        con.close()
        scratch.unlink(missing_ok=True)
        Path(str(scratch) + ".wal").unlink(missing_ok=True)
    tables = {"residual_cells.csv": cells,
              "residual_floor_areas.csv": areas,
              "residual_rebuild_costs.csv": costs_table.drop(columns="region_name"),
              "tpa_residual_reference.csv": parse_tpa_cells(paths["tpa_lvt_model_cells"])}
    table_meta = {}
    for name, frame in tables.items():
        path = DERIVED_DIR / name
        frame.to_csv(path, index=False, float_format="%.12f", lineterminator="\n")
        table_meta[name] = dict(sha256=sha256_file(path), rows=len(frame))
    sources, _ = load_manifest()
    meta = dict(schema_version=1, valuation_date=VALUATION_DATE,
                sources={sid: sources[sid].sha256 for sid in SOURCE_IDS}, tables=table_meta,
                prototype_tables={str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in
                                  [DERIVED_DIR / "voa_ctsop1_2026_england_la.csv", REPO_ROOT / "results/land/la_lambda_raw.csv"]},
                coverage=coverage,
                ons=dict(latest_edition="Preliminary UK national balance sheet estimates, 10 June 2026",
                         latest_dwelling_land_status="not separately published; request required",
                         latest_uk_total_land_2025_gbp=6872455e6,
                         historical_uk_dwelling_land_gbp=parse_ons_dwelling_land(paths["ons_dwelling_land_2021"]),
                         historical_dwelling_land_date="2021-12-31", historical_precision_gbp=1e11,
                         historical_source_id="ons_dwelling_land_2021"))
    RESIDUAL_MANIFEST.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(coverage, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    _, top = load_manifest()
    derive(args.cache_dir or default_cache_dir(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
