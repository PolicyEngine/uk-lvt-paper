"""Residual arithmetic, stock allocation, geography and reproducible outputs."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from uk_lvt.land import DERIVED_DIR
from uk_lvt.land import residual as R
from uk_lvt.land import residual_derive as D
from uk_lvt.land import residual_inputs as I
from uk_lvt.land.fetch import HashMismatch, sha256_file
from uk_lvt.land.lambda_raw import convex_weights, weighted_mean


# Disable wall-clock deadlines on the shared host; these check mathematical
# properties rather than performance. Generated values cover plausible data
# with ample margin, and explicit tests cover invalid and extreme inputs.
TIMING_FREE = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
PRICES = st.floats(min_value=1.0, max_value=1e9, allow_nan=False)
AREAS = st.floats(min_value=0.0, max_value=1e5, allow_nan=False)
COSTS = st.floats(min_value=0.0, max_value=1e7, allow_nan=False)


@TIMING_FREE
@given(price=PRICES, area=AREAS, cost=COSTS)
def test_residual_share_is_bounded_and_matches_the_residual(price, area, cost):
    share = R.residual_share(price, area, cost)
    assert 0.0 <= share <= 1.0
    assert math.isclose(share, max(0.0, (price - area * cost) / price),
                        rel_tol=1e-12, abs_tol=1e-12)


@TIMING_FREE
@given(price=PRICES, area=AREAS, cost1=COSTS, cost2=COSTS)
def test_higher_rebuild_cost_cannot_increase_land_share(price, area, cost1, cost2):
    low, high = sorted((cost1, cost2))
    assert R.residual_share(price, area, high) <= R.residual_share(price, area, low)


def test_clipped_and_zero_cost_boundary_cases():
    assert R.residual_share(100_000, 100, 2_000) == 0.0
    assert R.residual_share(100_000, 100, 1_000) == 0.0
    assert R.residual_share(100_000, 100, 0) == 1.0
    assert R.residual_share(100_000, 0, 2_000) == 1.0
    # The ratio of two finite inputs can overflow before multiplication by zero.
    assert R.residual_share(1e-300, 1e300, 0) == 1.0


@pytest.mark.parametrize("price, area, cost", [
    (0, 100, 1_000), (-1, 100, 1_000), (float("nan"), 100, 1_000),
    (float("inf"), 100, 1_000), (100_000, -1, 1_000),
    (100_000, float("nan"), 1_000), (100_000, float("inf"), 1_000),
    (100_000, 100, -1), (100_000, 100, float("nan")),
    (100_000, 100, float("inf")),
])
def test_invalid_residual_inputs_are_rejected(price, area, cost):
    with pytest.raises(ValueError):
        R.residual_share(price, area, cost)


@TIMING_FREE
@given(st.lists(st.tuples(PRICES, AREAS, COSTS,
                         st.floats(min_value=0.001, max_value=1e7, allow_nan=False)),
                min_size=1, max_size=25))
def test_stock_aggregation_is_a_convex_combination(properties):
    shares = np.array([R.residual_share(p, a, c) for p, a, c, _ in properties])
    stock = np.array([w for _, _, _, w in properties])
    weights = convex_weights(stock)
    aggregate = weighted_mean(shares, stock)
    assert np.all(weights >= 0)
    assert math.isclose(weights.sum(), 1.0, rel_tol=1e-12)
    assert math.isclose(aggregate, float(np.dot(weights, shares)),
                        rel_tol=1e-12, abs_tol=1e-12)
    assert shares.min() - 1e-12 <= aggregate <= shares.max() + 1e-12


@pytest.fixture
def universe():
    return I.load_authority_universe()


@pytest.fixture
def synthetic_cells(universe):
    """Two unequally weighted strata per real LA, with independently known means."""
    records = []
    for row in universe.itertuples(index=False):
        for share, fraction, band in [(0.1, 0.25, "A"), (0.8, 0.75, "B")]:
            records.append(dict(
                la_code=row.la_code, property_type="F", band=band,
                stock=row.all_properties * fraction, mean_share=share,
                mean_share_cost_low=min(1.0, share + 0.1),
                mean_share_cost_high=max(0.0, share - 0.1),
                mean_price=100_000, mean_land=100_000 * share,
                sales=10, sales_clipped=0, fallback="observed",
            ))
    return pd.DataFrame(records)


def test_authority_aggregation_uses_stock_weights_and_keeps_296_once(universe, synthetic_cells):
    out = R.authority_table(synthetic_cells, universe)
    assert len(out) == 296
    assert out.la_code.is_unique
    assert set(out.la_code) == set(universe.la_code)
    np.testing.assert_allclose(out.lambda_residual, 0.625)
    np.testing.assert_allclose(out.stock_mean_land_gbp, 62_500)
    np.testing.assert_allclose(out.implied_land_gbp, out.dwellings * 62_500)
    assert out.lambda_residual.between(0.1, 0.8).all()
    assert (out.lambda_cost_plus25 <= out.lambda_residual).all()
    assert (out.lambda_cost_minus25 >= out.lambda_residual).all()


def test_authority_aggregation_rejects_missing_stock(universe, synthetic_cells):
    incomplete = synthetic_cells.iloc[1:]
    with pytest.raises(ValueError, match="stock does not cover authority"):
        R.authority_table(incomplete, universe)


@pytest.fixture
def pinned_inputs(tmp_path, monkeypatch):
    """Small independent raw, derived and prototype bytes with their true hashes."""
    cache = tmp_path / "cache"
    derived = tmp_path / "derived"
    cache.mkdir()
    derived.mkdir()
    raw = cache / "source.csv"
    table = derived / "cells.csv"
    prototype = tmp_path / "prototype.csv"
    raw.write_text("source,value\noriginal,42\n")
    table.write_text("la_code,stock\nE06000001,1\n")
    prototype.write_text("reference,share\noriginal,0.5\n")
    digest = sha256_file(raw)
    source = SimpleNamespace(sha256=digest, filename=raw.name)
    meta = dict(
        sources={"fixture_source": digest},
        tables={table.name: {"sha256": sha256_file(table)}},
        prototype_tables={prototype.name: sha256_file(prototype)},
    )
    manifest = derived / "manifest.json"
    manifest.write_text(json.dumps(meta))
    monkeypatch.setattr(R, "DERIVED_DIR", derived)
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(R, "RESIDUAL_MANIFEST", manifest)
    monkeypatch.setattr(R, "load_manifest", lambda: (
        {"fixture_source": source}, {"default_cache_dir": "cache"}))
    return dict(cache=cache, raw=raw, derived=table, prototype=prototype,
                source=source, meta=meta)


def test_manifest_verifies_pins_and_cached_and_derived_bytes(pinned_inputs):
    assert R.verify_inputs(cache_dir=pinned_inputs["cache"]) == pinned_inputs["meta"]


@pytest.mark.parametrize("kind", ["raw", "derived", "prototype"])
def test_manifest_detects_tampering_in_each_input_layer(pinned_inputs, kind):
    pinned_inputs[kind].write_text("tampered\n")
    with pytest.raises(HashMismatch):
        R.verify_inputs(cache_dir=pinned_inputs["cache"])


def test_manifest_rejects_source_pin_disagreement(pinned_inputs):
    pinned_inputs["source"].sha256 = "0" * 64
    with pytest.raises(HashMismatch, match="provenance differs"):
        R.verify_inputs(cache_dir=pinned_inputs["cache"])


def test_deleted_raw_cache_still_allows_verified_offline_inputs(pinned_inputs):
    pinned_inputs["raw"].unlink()
    assert R.verify_inputs(cache_dir=pinned_inputs["cache"]) == pinned_inputs["meta"]


def test_real_input_manifest_hashes_match_without_requiring_raw_cache(tmp_path):
    # Offline portability requires the small derived files and source pins;
    # deleting large, reproducibly downloadable raw files is permitted.
    meta = R.verify_inputs(cache_dir=tmp_path)
    assert meta["sources"]
    assert "residual_cells.csv" in meta["tables"]


def test_committed_authority_cells_are_convex_and_cover_each_la_once(universe):
    cells = pd.read_csv(DERIVED_DIR / "residual_cells.csv", keep_default_na=False)
    assert not cells.duplicated(["la_code", "property_type", "band"]).any()
    assert (cells.stock >= 0).all()
    assert cells.mean_share.between(0, 1).all()
    assert (cells.mean_share_cost_high <= cells.mean_share + 1e-12).all()
    assert (cells.mean_share_cost_low >= cells.mean_share - 1e-12).all()
    out = R.authority_table(cells, universe).set_index("la_code")
    assert len(out) == 296 and out.index.is_unique
    assert set(out.index) == set(universe.la_code)
    for code, group in cells.groupby("la_code"):
        shares, stock = group.mean_share.to_numpy(), group.stock.to_numpy()
        expected = np.dot(convex_weights(stock), shares)
        assert math.isclose(out.loc[code, "lambda_residual"], expected,
                            rel_tol=1e-12, abs_tol=1e-12)
        assert shares.min() - 1e-12 <= expected <= shares.max() + 1e-12


def test_offline_outputs_reproduce_committed_bytes():
    payloads = R.output_payloads()
    assert len(payloads) == 4
    for path, content in payloads.items():
        assert path.read_bytes() == content.encode("utf-8"), path.name


@pytest.mark.parametrize("district, expected", [
    ("Allerdale", "E06000063"), ("Copeland", "E06000063"),
    ("Barrow-in-Furness", "E06000064"), ("South Lakeland", "E06000064"),
    ("Scarborough", "E06000065"), ("York", "E06000014"),
    ("Taunton Deane", "E06000066"), ("Somerset West and Taunton", "E06000066"),
    ("Bath and North East Somerset", "E06000022"), ("North Somerset", "E06000024"),
    ("Christchurch", "E06000058"), ("Weymouth and Portland", "E06000059"),
    ("Wycombe", "E06000060"), ("Milton Keynes", "E06000042"),
    ("Corby", "E06000061"), ("Daventry", "E06000062"),
    ("Waveney", "E07000244"), ("St Edmundsbury", "E07000245"),
    ("Shepway", "E07000112"), ("CITY OF BRISTOL", "E06000023"),
    ("Kingston upon Hull", "E06000010"), ("Barnsley", "E08000038"),
    ("Sheffield", "E08000039"), ("Isles of Scilly", "E06000053"),
])
def test_price_paid_historical_districts_use_2023_authorities(district, expected):
    assert I.ppd_district_to_la(district) == expected


def test_district_normalisation_is_exact_and_does_not_guess_geography():
    assert I.normalize_district_name("  King's-Lynn & West Norfolk  ") == "KINGS LYNN AND WEST NORFOLK"
    assert I.ppd_district_to_la("  King's Lynn & West Norfolk  ") == "E07000146"
    assert I.ppd_district_to_la("Cardiff") is None
    assert I.ppd_district_to_la("Unrecognised English district") is None
    assert I.ppd_district_to_la("SW1A 1AA") is None


def _write_stock_ods(path: Path, bands=I.BANDS, duplicate_a=False):
    """Minimal source-shaped ODS, including huge repeated blank-cell runs."""
    table_ns = I._TABLE_NS
    text_ns = I._TEXT_NS
    root = ET.Element("document")
    sheet = ET.SubElement(root, f"{{{table_ns}}}table", {f"{{{table_ns}}}name": "CTSOP3_0"})
    header = ["Geography", "Name", "Code", "Unused", "Band", "Detached House",
              "Semi-Detached House", "Terraced House", "Flat / Maisonette", "Bungalow",
              "Annexe", "Caravan/ Houseboat/ Mobile Home", "Unknown [note 17]"]
    rows = [header]
    for band in bands:
        counts = ["10", "30", "[z]", "10", "20", "[c]", "[z]", "5"] if band == "A" else (
            ["[z]"] * 7 + ["10"] if band == "B" else ["[z]"] * 8)
        row = ["LAUA", "Test authority", "E06000001", "", band, *counts]
        rows.append(row)
        if duplicate_a and band == "A":
            rows.append(row)
    for values in rows:
        element = ET.SubElement(sheet, f"{{{table_ns}}}table-row")
        for value in values:
            cell = ET.SubElement(element, f"{{{table_ns}}}table-cell")
            ET.SubElement(cell, f"{{{text_ns}}}p").text = value
        ET.SubElement(element, f"{{{table_ns}}}table-cell", {
            f"{{{table_ns}}}number-columns-repeated": "1000000"})
    with ZipFile(path, "w") as archive:
        archive.writestr("content.xml", ET.tostring(root))
    return pd.DataFrame([dict(la_code="E06000001", all_properties=175)])


def test_voa_stock_suppression_allocation_and_total_reconciliation(tmp_path):
    path = tmp_path / "stock.ods"
    universe = _write_stock_ods(path)
    stock = I.parse_ctsop3_stock(path, universe)
    assert len(stock) == 32
    assert not stock.duplicated(["la_code", "property_type", "band"]).any()
    assert set(stock.property_type) == set(I.PROPERTY_TYPES)
    assert set(stock.band) == set(I.BANDS)
    assert (stock.stock >= 0).all()
    assert math.isclose(stock.stock.sum(), 175)
    assert stock.attrs["suppressed_cells_imputed"] == 1
    assert stock.attrs["suppressed_count_imputation"] == 2.5
    assert stock.attrs["minimum_authority_scale_factor"] == 2.0
    # Band B has only unknown stock: it uses the independent authority mix.
    b = stock[stock.band.eq("B")].set_index("property_type").stock
    np.testing.assert_allclose(b.loc[["D", "S", "T", "F"]], [30 / 7, 90 / 7, 0, 20 / 7])
    # Band A's 20 bungalows follow its 1:3 detached/semi house mix.
    a = stock[stock.band.eq("A")].set_index("property_type").stock
    assert math.isclose(a["S"] / a["D"], 3.0)
    assert a["T"] == 0


def test_voa_stock_rejects_duplicate_and_missing_bands(tmp_path):
    duplicate = tmp_path / "duplicate.ods"
    universe = _write_stock_ods(duplicate, duplicate_a=True)
    with pytest.raises(ValueError, match="duplicate CTSOP3.0 authority-band"):
        I.parse_ctsop3_stock(duplicate, universe)
    missing = tmp_path / "missing.ods"
    _write_stock_ods(missing, bands=I.BANDS[:-1])
    with pytest.raises(ValueError, match="missing bands"):
        I.parse_ctsop3_stock(missing, universe)


def test_voa_stock_rejects_missing_authority_and_nonpositive_totals(tmp_path):
    path = tmp_path / "stock.ods"
    universe = _write_stock_ods(path)
    expanded = pd.concat([universe, universe.assign(la_code="E06000002")], ignore_index=True)
    with pytest.raises(ValueError, match="authority mismatch"):
        I.parse_ctsop3_stock(path, expanded)
    with pytest.raises(ValueError, match="nonpositive dwelling total"):
        I.parse_ctsop3_stock(path, universe.assign(all_properties=0))


@pytest.mark.parametrize("invalid", ["[x]", "..", "", "-1", "1.5"])
def test_unknown_voa_suppression_markers_are_rejected(invalid):
    with pytest.raises(ValueError, match="unexpected CTSOP3.0 stock cell"):
        I._stock_count(invalid)


@TIMING_FREE
@given(prices=st.lists(st.one_of(PRICES, st.sampled_from([100.0, 200.0, 300.0])),
                      min_size=1, max_size=60),
       counts=st.lists(st.integers(min_value=0, max_value=10_000),
                       min_size=8, max_size=8))
def test_proxy_bands_are_nondecreasing_in_price_and_deterministic(prices, counts):
    assume(sum(counts) > 0)
    assigned = D.assign_proxy_bands(prices, counts)
    assert len(assigned) == len(prices)
    assert set(assigned) <= set(I.BANDS)
    order = np.argsort(prices, kind="stable")
    assert list(assigned[order]) == sorted(assigned)
    np.testing.assert_array_equal(assigned, D.assign_proxy_bands(prices, counts))
    # Stock count units do not alter their quantile thresholds.
    np.testing.assert_array_equal(assigned, D.assign_proxy_bands(prices, np.asarray(counts) * 7))


def test_proxy_bands_preserve_input_rows_and_exact_quantile_multiplicities():
    prices = np.array([80, 10, 60, 30, 20, 70, 40, 50])
    assigned = D.assign_proxy_bands(prices, [1] * 8)
    np.testing.assert_array_equal(assigned, list("HAFCBGDE"))
    assert dict(zip(*np.unique(assigned, return_counts=True))) == dict.fromkeys(I.BANDS, 1)
    repeated = D.assign_proxy_bands(np.arange(1, 17), [1] * 8)
    np.testing.assert_array_equal(repeated, np.repeat(list(I.BANDS), 2))
    # Zero-stock bands receive no sales; the two populated intervals are exact.
    uneven = D.assign_proxy_bands(np.arange(1, 9), [2, 0, 6, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(uneven, list("AACCCCCC"))


def test_proxy_band_boundary_belongs_to_the_next_interval_and_ties_are_stable():
    # n=2 midpoint rank 0.25 lies exactly on A's upper threshold (1/4).
    np.testing.assert_array_equal(D.assign_proxy_bands([1, 2], [1, 3, 0, 0, 0, 0, 0, 0]), ["B", "B"])
    # Equal prices retain input order, allowing an explicit reproducible split.
    ties = np.repeat(100_000.0, 8)
    np.testing.assert_array_equal(D.assign_proxy_bands(ties, [1] * 8), list(I.BANDS))
    assert D.assign_proxy_bands([], [1] * 8).shape == (0,)


@pytest.mark.parametrize("prices, counts", [
    ([1], [1] * 7), ([1], [0] * 8), ([1], [-1] + [1] * 7),
    ([1], [float("nan")] + [1] * 7), ([1], [float("inf")] + [1] * 7),
    ([0], [1] * 8), ([-1], [1] * 8),
    ([float("nan")], [1] * 8), ([float("inf")], [1] * 8),
])
def test_invalid_proxy_band_inputs_are_rejected(prices, counts):
    with pytest.raises(ValueError):
        D.assign_proxy_bands(prices, counts)


@pytest.fixture
def sparse_inputs():
    universe = pd.DataFrame([
        dict(la_code="E06000001", region_name="North East"),
        dict(la_code="E06000002", region_name="North East"),
        dict(la_code="E06000003", region_name="North East"),
        dict(la_code="E06000004", region_name="North West"),
    ])
    specs = [
        ("E06000001", "D", "A", 10, 2, 100_000, 0.2),
        ("E06000001", "D", "B", 30, 3, 300_000, 0.6),
        ("E06000003", "D", "A", 20, 0, 500_000, 0.9),
        # This large pool in another region must not contaminate North East.
        ("E06000004", "D", "A", 1_000, 900, 10_000, 0.01),
        ("E06000003", "F", "A", 5, 1, 150_000, 0.4),
    ]
    observed = pd.DataFrame([
        dict(la_code=code, property_type=kind, band=band, sales=n,
             sales_clipped=clipped, mean_price=price, mean_land=price * share,
             mean_share=share, mean_share_cost_low=min(1.0, share + 0.1),
             mean_share_cost_high=max(0.0, share - 0.1))
        for code, kind, band, n, clipped, price, share in specs
    ])
    stock = observed[["la_code", "property_type", "band"]].assign(stock=10.0)
    missing = pd.DataFrame([
        dict(la_code="E06000001", property_type="D", band="C", stock=80.0),
        dict(la_code="E06000002", property_type="D", band="H", stock=100.0),
        dict(la_code="E06000002", property_type="F", band="H", stock=50.0),
        dict(la_code="E06000002", property_type="D", band="B", stock=0.0),
    ])
    return pd.concat([stock, missing], ignore_index=True), observed, universe


def test_sparse_fallbacks_use_la_type_then_region_type_sale_weighted_means(sparse_inputs):
    stock, observed, universe = sparse_inputs
    result = D.fill_sparse_cells(stock, observed, universe)
    indexed = result.set_index(["la_code", "property_type", "band"])
    local = indexed.loc[("E06000001", "D", "C")]
    regional = indexed.loc[("E06000002", "D", "H")]
    flat = indexed.loc[("E06000002", "F", "H")]
    assert local.fallback == "la_type"
    assert regional.fallback == flat.fallback == "region_type"
    la_donors = observed[observed.la_code.eq("E06000001") & observed.property_type.eq("D")]
    region_donors = observed[observed.la_code.isin(["E06000001", "E06000003"]) & observed.property_type.eq("D")]
    flat_donor = observed[observed.property_type.eq("F")].iloc[0]
    for col in D.STAT_COLUMNS:
        assert math.isclose(local[col], np.average(la_donors[col], weights=la_donors.sales))
        assert math.isclose(regional[col], np.average(region_donors[col], weights=region_donors.sales))
        assert flat[col] == flat_donor[col]
    assert math.isclose(local.mean_share, 0.5)
    assert math.isclose(local.mean_land, 140_000)
    assert math.isclose(regional.mean_share, 38 / 60)
    assert local.mean_share != local.mean_land / local.mean_price


def test_sparse_fallbacks_preserve_real_sales_counts_clips_stock_and_observed_values(sparse_inputs):
    stock, observed, universe = sparse_inputs
    result = D.fill_sparse_cells(stock, observed, universe)
    assert len(result) == len(stock)
    assert not result.duplicated(["la_code", "property_type", "band"]).any()
    assert result.sales.sum() == observed.sales.sum() == 1_065
    assert result.sales_clipped.sum() == observed.sales_clipped.sum() == 906
    fallback = result[result.fallback.ne("observed")]
    assert fallback.stock.gt(0).any()
    assert fallback.sales.eq(0).all()
    assert fallback.sales_clipped.eq(0).all()
    keys = ["la_code", "property_type", "band"]
    pd.testing.assert_frame_equal(result[keys + ["stock"]].sort_values(keys).reset_index(drop=True),
                                  stock.sort_values(keys).reset_index(drop=True))
    actual = result[result.fallback.eq("observed")][observed.columns].sort_values(keys).reset_index(drop=True)
    expected = observed.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_sparse_fallbacks_reject_duplicate_source_cells(sparse_inputs):
    stock, observed, universe = sparse_inputs
    duplicated = pd.concat([observed, observed.iloc[:1]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        D.fill_sparse_cells(stock, duplicated, universe)
