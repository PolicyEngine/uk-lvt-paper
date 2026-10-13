"""Exercise the real CSV/SQL ingestion path with small independent deeds."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uk_lvt.land import DERIVED_DIR, REPO_ROOT
from uk_lvt.land import residual_derive as D
from uk_lvt.land import residual_inputs as I
from uk_lvt.land import residual_reference as R
from uk_lvt.land.manifest import load_manifest


@pytest.fixture
def connection():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(config={"threads": 2, "memory_limit": "128MB"})
    try:
        yield con
    finally:
        con.close()


def deed(transaction_id, **changes):
    row = dict(zip(D.PPD_COLUMNS, [
        transaction_id, "100000", "2020-03-17 00:00", "EX1 1AA", "F", "N", "L",
        "42", "", "EXAMPLE STREET", "", "UNRELATED POSTAL TOWN",
        "CITY OF WESTMINSTER", "UNRELATED COUNTY", "A", "A",
    ]))
    row.update(changes)
    return row


def register_deeds(connection, path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, quoting=csv.QUOTE_ALL)
        writer.writerows([[row[name] for name in D.PPD_COLUMNS] for row in rows])
    D.read_price_paid(connection, [str(path)])
    mapping = I.ppd_district_crosswalk()
    districts = [row[0] for row in connection.execute("select distinct district from raw").fetchall()]
    connection.register("crosswalk", pd.DataFrame([
        dict(district=name, la_code=I.ppd_district_to_la(name, mapping)) for name in districts
    ]))


def test_latest_address_sale_uses_date_then_id_and_preserves_separate_flats(connection, tmp_path):
    register_deeds(connection, tmp_path / "deeds.csv", [
        deed("old", sale_date="2018-01-01 00:00", price="90000"),
        deed("newer", sale_date="2024-05-31 00:00", price="120000", record_status="C"),
        deed("zzzz-older", sale_date="2020-03-01 00:00", price="95000"),
        deed("tie-a", sale_date="2025-06-30 00:00", price="130000"),
        deed("tie-z", sale_date="2025-06-30 00:00", price="140000"),
        deed("flat-1", paon="43", saon="FLAT 1", price="110000"),
        deed("flat-2", paon="43", saon="FLAT 2", price="160000"),
    ])
    D.create_latest_sales(connection)
    selected = connection.execute(
        "select transaction_id,price,month,la_code from latest order by transaction_id"
    ).fetchall()
    assert selected == [
        ("flat-1", 110000.0, "2020-03-01", "E09000033"),
        ("flat-2", 160000.0, "2020-03-01", "E09000033"),
        ("tie-z", 140000.0, "2025-06-01", "E09000033"),
    ]
    assert connection.execute("select count(distinct property_key) from latest").fetchone()[0] == 3


def test_eligible_categories_statuses_types_addresses_and_dates(connection, tmp_path):
    rejected = [
        dict(category="B"), dict(record_status="D"), dict(record_status="X"),
        dict(property_type="O"), dict(price="0"), dict(postcode=""), dict(paon=""),
        dict(sale_date="2017-12-31 00:00"), dict(sale_date="2026-05-01 00:00"),
        dict(district="CARDIFF"),
    ]
    rows = [deed(f"reject-{i}", **({"paon": str(i + 10)} | changes))
            for i, changes in enumerate(rejected)]
    rows += [
        deed("accept-A", paon="101", sale_date="2018-01-01 00:00"),
        deed("accept-C", paon="102", sale_date="2026-04-30 00:00", record_status="C"),
    ]
    register_deeds(connection, tmp_path / "deeds.csv", rows)
    D.create_latest_sales(connection)
    selected = connection.execute("select transaction_id from latest order by transaction_id").fetchall()
    assert selected == [("accept-A",), ("accept-C",)]


def test_geography_is_the_district_field_and_hpi_joins_exact_authority_month(connection, tmp_path):
    register_deeds(connection, tmp_path / "deeds.csv", [
        deed("westminster", paon="1", price="100000", sale_date="2019-03-10 00:00",
             town="CARDIFF", county="ALLERDALE"),
        deed("cumberland", paon="2", district="ALLERDALE", price="80000",
             sale_date="2020-05-20 00:00", town="CITY OF WESTMINSTER"),
        deed("wrekin", paon="3", district="WREKIN", price="120000",
             sale_date="2021-01-05 00:00"),
    ])
    D.create_latest_sales(connection)
    connection.register("hpi", pd.DataFrame([
        dict(la_code="E09000033", month="2019-03-01", hpi=100.0),
        dict(la_code="E09000033", month="2020-05-01", hpi=999.0),
        dict(la_code="E06000063", month="2020-05-01", hpi=80.0),
        dict(la_code="E06000063", month="2019-03-01", hpi=777.0),
        dict(la_code="E06000020", month="2021-01-01", hpi=120.0),
    ]))
    connection.register("target", pd.DataFrame([
        dict(la_code="E09000033", target_hpi=200.0),
        dict(la_code="E06000063", target_hpi=120.0),
        dict(la_code="E06000020", target_hpi=90.0),
    ]))
    connection.register("rebuild", pd.DataFrame([
        dict(la_code="E09000033", property_type="F", rebuild_cost=130000.0),
        dict(la_code="E09000033", property_type="D", rebuild_cost=500000.0),
        dict(la_code="E06000063", property_type="F", rebuild_cost=90000.0),
        dict(la_code="E06000020", property_type="F", rebuild_cost=100000.0),
    ]))
    D.create_priced_sales(connection)
    selected = connection.execute(
        "select transaction_id,la_code,value,cost from priced order by transaction_id"
    ).fetchall()
    assert selected == [
        ("cumberland", "E06000063", 120000.0, 90000.0),
        ("westminster", "E09000033", 200000.0, 130000.0),
        ("wrekin", "E06000020", 90000.0, 100000.0),
    ]


def test_price_rank_bands_skip_zero_stock_bands_and_ties_are_deterministic():
    # C is empty. Midpoint ranks enter A, B, D and H at exact known cutoffs.
    prices = [400000, 100000, 100000, 300000]
    counts = [1, 1, 0, 1, 0, 0, 0, 1]
    assert D.assign_proxy_bands(prices, counts).tolist() == ["H", "A", "B", "D"]


def test_sql_ranks_and_residual_statistics_match_independent_arithmetic(connection):
    connection.register("price_rows", pd.DataFrame([
        dict(la_code="E06000001", property_type="F", value=price, cost=200.0,
             property_key=str(price)) for price in [500.0, 100.0, 400.0, 300.0]
    ] + [dict(la_code="E06000002", property_type="D", value=500.0, cost=100.0,
              property_key="different-authority")]))
    connection.execute("create table priced as select * from price_rows")
    connection.register("bands", pd.DataFrame([
        dict(la_code="E06000001", band="A", lower=0.0, upper=.25),
        dict(la_code="E06000001", band="B", lower=.25, upper=.75),
        dict(la_code="E06000001", band="C", lower=.75, upper=.75),
        dict(la_code="E06000001", band="H", lower=.75, upper=1.0),
        dict(la_code="E06000002", band="A", lower=0.0, upper=1.0),
    ]))
    D.create_ranked_sales(connection)
    ranks = connection.execute('select "quantile" from ranked where la_code=\'E06000001\' order by "value"').fetchall()
    assert ranks == [(.125,), (.375,), (.625,), (.875,)]
    cells = D.observed_cell_statistics(connection).set_index(["la_code", "band"])
    assert cells.sales.sum() == 5
    assert cells.sales_clipped.sum() == 1
    assert ("E06000001", "C") not in cells.index
    clipped, pooled = cells.loc[("E06000001", "A")], cells.loc[("E06000001", "B")]
    assert clipped.mean_land == clipped.mean_share == 0.0
    assert clipped.sales_clipped == 1
    assert pooled.sales == 2
    assert pooled.mean_price == 350.0
    assert pooled.mean_land == 150.0
    assert pooled.mean_share == pytest.approx(5 / 12)
    assert pooled.mean_share_cost_low == pytest.approx(.5625)
    assert pooled.mean_share_cost_high == pytest.approx(13 / 48)
    assert cells.loc[("E06000002", "A"), "mean_share"] == .8


def test_sparse_stock_cells_use_sale_count_weighted_local_then_region_donors():
    stock = pd.DataFrame([
        dict(la_code="E06000001", property_type="F", band=band, stock=10.0)
        for band in "ABC"
    ] + [dict(la_code="E06000002", property_type="F", band="A", stock=10.0)])
    observed = pd.DataFrame([
        dict(la_code="E06000001", property_type="F", band="A", sales=1, sales_clipped=0,
             mean_price=100.0, mean_land=10.0, mean_share=.1,
             mean_share_cost_low=.2, mean_share_cost_high=0.0),
        dict(la_code="E06000001", property_type="F", band="B", sales=3, sales_clipped=0,
             mean_price=300.0, mean_land=210.0, mean_share=.7,
             mean_share_cost_low=.8, mean_share_cost_high=.6),
    ])
    universe = pd.DataFrame([
        dict(la_code=code, region_name="North East") for code in ["E06000001", "E06000002"]
    ])
    filled = D.fill_sparse_cells(stock, observed, universe).set_index(["la_code", "band"])
    local, regional = filled.loc[("E06000001", "C")], filled.loc[("E06000002", "A")]
    assert local.fallback == "la_type"
    assert regional.fallback == "region_type"
    for row in [local, regional]:
        assert row.mean_price == 250.0
        assert row.mean_land == 160.0
        assert row.mean_share == pytest.approx(.55)
        assert row.sales == row.sales_clipped == 0


def test_ons_parser_accepts_source_whitespace_and_converts_trillions(tmp_path):
    path = tmp_path / "ons.csv"
    path.write_text('"Unit","£ trillion\t"\n'
                    '"Year ","Land underlying dwellings\t"\n'
                    '"2020","5.4"\n"2021\t","5.9 "\n')
    assert R.parse_ons_dwelling_land(path) == 5.9e12
    assert R.parse_ons_dwelling_land(path, year=2020) == 5.4e12
    with pytest.raises(ValueError, match="no observation"):
        R.parse_ons_dwelling_land(path, year=2022)
    path.write_text('"Unit","£ million"\n"Year","Land underlying dwellings"\n"2021","5900000"\n')
    with pytest.raises(ValueError, match="£ trillion"):
        R.parse_ons_dwelling_land(path)


def cached_source(source_id):
    sources, top = load_manifest()
    path = REPO_ROOT / top["default_cache_dir"] / sources[source_id].filename
    if not path.exists():
        pytest.skip(f"optional pinned raw cache absent: {source_id}")
    return path


def test_cached_ehs_floor_areas_reproduce_small_committed_table():
    raw = cached_source("ehs_2018_floor_areas")
    committed = DERIVED_DIR / "residual_floor_areas.csv"
    if not committed.exists():
        pytest.skip("residual floor areas not derived yet")
    actual = R.parse_floor_areas(raw).sort_values(["region_name", "property_type"]).reset_index(drop=True)
    expected = pd.read_csv(committed).sort_values(["region_name", "property_type"]).reset_index(drop=True)
    assert len(actual) == 36
    assert actual[["region_name", "property_type"]].equals(expected[["region_name", "property_type"]])
    np.testing.assert_allclose(actual.floor_area_m2, expected.floor_area_m2, rtol=1e-12)


def test_cached_tpa_constants_and_regions_reproduce_small_committed_cost_table():
    costs, factors = R.parse_rebuild_constants(cached_source("tpa_rebuild_config"))
    assert costs == {"D": 2100.0, "S": 1950.0, "T": 1900.0, "F": 2400.0}
    committed = DERIVED_DIR / "residual_rebuild_costs.csv"
    if not committed.exists():
        pytest.skip("residual rebuilding costs not derived yet")
    expected = pd.read_csv(committed)
    cost_per_m2 = expected.property_type.map(costs) * expected.region_code.map(factors)
    assert len(expected) == 296 * 4
    assert not expected.duplicated(["la_code", "property_type"]).any()
    np.testing.assert_allclose(expected.cost_per_m2_gbp, cost_per_m2, rtol=1e-12)
    np.testing.assert_allclose(expected.rebuild_cost, expected.floor_area_m2 * cost_per_m2, rtol=1e-12)


def test_cached_ons_fallback_matches_the_published_rounded_number():
    assert R.parse_ons_dwelling_land(cached_source("ons_dwelling_land_2021")) == 5.9e12
