"""Data tests for the committed land tables and results (no downloads needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from uk_lvt.land import (
    DERIVED_DIR,
    DERIVED_MANIFEST_PATH,
    METHODS_NOTE_PATH,
    RESULTS_DIR,
)
from uk_lvt.land import lambda_raw as L
from uk_lvt.land.derive import CTSOP_FILE, HPI_FILE, MHCLG_FILE
from uk_lvt.land.fetch import sha256_file
from uk_lvt.land.geography import (
    CORNWALL,
    ENGLAND_CODE,
    ISLES_OF_SCILLY,
    MHCLG_TO_2023,
    N_ENGLISH_AUTHORITIES,
)
from uk_lvt.land.manifest import load_manifest

CUMBERLAND = "E06000063"


@pytest.fixture(scope="module")
def la():
    return pd.read_csv(RESULTS_DIR / L.LA_FILE, dtype={"la_code": str, "mhclg_code": str},
                       keep_default_na=True)


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(RESULTS_DIR / L.SUMMARY_FILE)


@pytest.fixture(scope="module")
def ctsop():
    return pd.read_csv(DERIVED_DIR / CTSOP_FILE, dtype={"la_code": str})


@pytest.fixture(scope="module")
def mhclg():
    return pd.read_csv(DERIVED_DIR / MHCLG_FILE, dtype={"la_code_mhclg": str})


# --------------------------------------------------------------------------
# All 296 authorities present exactly once
# --------------------------------------------------------------------------

def test_all_296_authorities_present_exactly_once(la, ctsop):
    assert len(la) == N_ENGLISH_AUTHORITIES
    assert la["la_code"].is_unique
    assert set(la["la_code"]) == set(ctsop["la_code"])
    assert len(ctsop) == N_ENGLISH_AUTHORITIES and ctsop["la_code"].is_unique
    assert la["la_code"].str.match(r"^E0[6-9]\d{6}$").all()
    assert la["la_code"].str[:3].value_counts().to_dict() == {"E07": 164, "E06": 63, "E08": 36, "E09": 33}


def test_pre_2023_codes_are_crosswalked(la):
    assert not set(MHCLG_TO_2023) & set(la["la_code"])
    s = la.set_index("la_code")
    for old, new in MHCLG_TO_2023.items():
        assert s.loc[new, "mhclg_code"] == old
        assert s.loc[new, "method_flag"] == "computed_code_crosswalk"
    assert (la["method_flag"] == "computed_code_crosswalk").sum() == len(MHCLG_TO_2023)


def test_isles_of_scilly_takes_cornwall(la):
    s = la.set_index("la_code")
    assert s.loc[ISLES_OF_SCILLY, "method_flag"] == f"imputed_from_{CORNWALL}"
    assert pd.isna(s.loc[ISLES_OF_SCILLY, "mhclg_code"])
    lam_cols = [c for c in la.columns if c.startswith("lambda_raw")]
    for col in lam_cols + ["stock_price", "plot_land_per_dwelling_gbp"]:
        assert s.loc[ISLES_OF_SCILLY, col] == s.loc[CORNWALL, col]
    assert la["method_flag"].str.startswith("imputed").sum() == 1


def test_inputs_cover_the_universe_as_documented(la, mhclg):
    hpi = pd.read_csv(DERIVED_DIR / HPI_FILE, dtype={"la_code": str})
    universe = set(la["la_code"])
    assert universe - set(hpi["la_code"]) == {ISLES_OF_SCILLY}
    mhclg_2023 = {MHCLG_TO_2023.get(c, c) for c in mhclg["la_code_mhclg"]}
    assert universe - mhclg_2023 == {ISLES_OF_SCILLY}
    assert mhclg_2023 <= universe


# --------------------------------------------------------------------------
# Bounds and sensitivity direction on the real data
# --------------------------------------------------------------------------

def test_lambda_columns_on_unit_interval(la):
    lam_cols = [c for c in la.columns if c.startswith("lambda_raw")]
    vals = la[lam_cols].to_numpy(dtype=float)
    finite = vals[~np.isnan(vals)]
    assert ((finite >= 0) & (finite <= 1)).all()
    assert la["lambda_raw"].notna().all()
    # Only the no-floor variant has a gap: Cumberland has no value at the median.
    missing = la.loc[la["lambda_raw_unadjusted"].isna(), "la_code"].tolist()
    assert missing == [CUMBERLAND]
    assert (la["capped_variants"].isna()).all()


def test_density_sensitivity_direction_on_data(la):
    assert (la["lambda_raw_density_plus25"] >= la["lambda_raw"]).all()
    assert (la["lambda_raw_density_minus25"] <= la["lambda_raw"]).all()
    np.testing.assert_allclose(la["lambda_raw_density_plus25"], 1.25 * la["lambda_raw"], atol=2e-6)
    np.testing.assert_allclose(la["lambda_raw_density_minus25"], 0.75 * la["lambda_raw"], atol=2e-6)
    assert (la["lambda_raw_price_p25"] <= la["lambda_raw"]).all()
    assert (la["lambda_raw"] <= la["lambda_raw_price_p75"]).all()


def test_plot_land_and_lambda_identities(la):
    np.testing.assert_allclose(
        la["plot_land_per_dwelling_gbp"],
        la["land_value_central_gbp_per_ha"] / la["density_central_dph"], atol=0.006)
    np.testing.assert_allclose(
        la["lambda_raw"], la["plot_land_per_dwelling_gbp"] / la["stock_price"], atol=2e-6)


# --------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------

def test_summary_matches_rows_and_is_bounded(la, summary):
    for (code, variant, weighting), row in summary.set_index(
            ["geography_code", "variant", "weighting"]).iterrows():
        g = la if code == ENGLAND_CODE else la[la["region_code"] == code]
        col = L.variant_column(variant)
        vals = g[col].to_numpy(dtype=float)
        w = {"dwellings": g["dwellings"].to_numpy(dtype=float),
             "value": g["dwellings"].to_numpy(dtype=float) * g["stock_price"].to_numpy(dtype=float),
             "unweighted": np.ones(len(g))}[weighting]
        keep = ~np.isnan(vals)
        expected = float(np.dot(w[keep], vals[keep]) / w[keep].sum())
        assert row["lambda_raw"] == pytest.approx(expected, abs=1e-6)
        assert np.nanmin(vals) - 1e-9 <= row["lambda_raw"] <= np.nanmax(vals) + 1e-9
        assert row["n_authorities"] == keep.sum()
        assert row["dwellings"] == g["dwellings"].sum()
    eng = summary[(summary["geography_code"] == ENGLAND_CODE)]
    assert set(eng["variant"]) == set(L.VARIANT_NAMES)
    assert summary["geography_code"].nunique() == 10  # England + 9 regions


# --------------------------------------------------------------------------
# Reproducibility and provenance
# --------------------------------------------------------------------------

def test_results_rebuild_byte_for_byte():
    built = L.build_outputs()
    for name, text in built.items():
        assert (RESULTS_DIR / name).read_text() == text, f"{name} is stale; run python -m uk_lvt.land.lambda_raw"


def test_methods_note_tables_match_results():
    la = L.build_la_table(*L.load_inputs())
    summary = L.summarise(la)
    note = METHODS_NOTE_PATH.read_text()
    assert L.replace_generated_blocks(note, la, summary) == note


def test_derived_tables_match_their_manifest_and_sources():
    derived = json.loads(DERIVED_MANIFEST_PATH.read_text())
    sources, _ = load_manifest()
    assert set(derived["files"]) == {MHCLG_FILE, HPI_FILE, CTSOP_FILE}
    for fname, meta in derived["files"].items():
        path = DERIVED_DIR / fname
        assert sha256_file(path) == meta["sha256"], f"{fname} edited without re-deriving"
        frame = pd.read_csv(path)
        assert len(frame) == meta["rows"]
        assert list(frame.columns) == meta["columns"]
        for src in meta["sources"]:
            assert sources[src["id"]].is_pinned
            assert sources[src["id"]].sha256 == src["sha256"]


# --------------------------------------------------------------------------
# MHCLG workbook invariants (guards against parsing errors)
# --------------------------------------------------------------------------

def test_mhclg_table_shape(mhclg):
    assert mhclg["la_code_mhclg"].nunique() == 295
    assert len(mhclg) == 295 * 2 * 3 * 5
    assert not mhclg.duplicated(["la_code_mhclg", "sheet", "density_scenario", "price_percentile"]).any()
    assert mhclg[mhclg["sheet"] == "floor_adjusted"]["land_value_gbp_per_ha"].notna().all()
    assert (mhclg["land_value_gbp_per_ha"].dropna() >= 0).all()


def test_mhclg_floor_never_below_unadjusted(mhclg):
    w = mhclg.pivot_table(index=["la_code_mhclg", "density_scenario", "price_percentile"],
                          columns="sheet", values="land_value_gbp_per_ha").dropna()
    assert (w["floor_adjusted"] >= w["unadjusted"]).all()
    assert len(w) == 4390 and int((w["floor_adjusted"] > w["unadjusted"]).sum()) == 115


def test_mhclg_values_rise_with_price_percentile(mhclg):
    for sheet, g in mhclg.groupby("sheet"):
        w = g.pivot_table(index=["la_code_mhclg", "density_scenario"], columns="price_percentile",
                          values="land_value_gbp_per_ha")
        diffs = w.diff(axis=1).iloc[:, 1:].to_numpy()
        assert (diffs[~np.isnan(diffs)] >= 0).all(), sheet


def test_mhclg_densities_within_published_bounds(mhclg):
    assert mhclg["dwellings_per_ha"].between(30, 400).all()
    assert (mhclg.groupby(["la_code_mhclg", "density_scenario"])["dwellings_per_ha"].nunique() == 1).all()
    d = mhclg[(mhclg["sheet"] == "floor_adjusted") & (mhclg["price_percentile"] == 50)].pivot_table(
        index="la_code_mhclg", columns="density_scenario", values="dwellings_per_ha")
    assert ((d["low"] <= d["medium"]) & (d["medium"] <= d["high"])).all()


def test_derived_tables_rebuild_from_cached_raw_files(tmp_path):
    """Runs only where the raw inputs are cached (python -m uk_lvt.land.fetch)."""
    pytest.importorskip("openpyxl")
    pytest.importorskip("odf")
    from uk_lvt.land import derive
    from uk_lvt.land.fetch import MissingInput

    try:
        derive.build_all(out_dir=tmp_path)
    except MissingInput as exc:
        pytest.skip(f"raw inputs not cached: {exc}")
    for fname in (MHCLG_FILE, HPI_FILE, CTSOP_FILE):
        assert (tmp_path / fname).read_bytes() == (DERIVED_DIR / fname).read_bytes()
    assert (tmp_path / DERIVED_MANIFEST_PATH.name).read_text() == DERIVED_MANIFEST_PATH.read_text()
