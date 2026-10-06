"""Property-based tests for the raw land-share arithmetic in uk_lvt.land.lambda_raw."""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from uk_lvt.land import lambda_raw as L
from uk_lvt.land.geography import CORNWALL, ISLES_OF_SCILLY, MHCLG_TO_2023

# Ranges cover the data with margin: MHCLG values reach about 1.9e8 GBP/ha,
# densities are 30-400 dph, UK HPI authority prices are about 1.1e5-1.5e6.
land_values = st.floats(min_value=0.0, max_value=1e9, allow_nan=False)
densities = st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False)
prices = st.floats(min_value=1e3, max_value=1e8, allow_nan=False)
factors = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)


# --------------------------------------------------------------------------
# 0 <= lambda <= 1
# --------------------------------------------------------------------------

@given(v=land_values, d=densities, p=prices, f=factors)
def test_lambda_in_unit_interval_and_exact_for_valid_inputs(v, d, p, f):
    ratio = v / (f * d) / p
    assume(ratio <= 1.0)  # valid: plot land per dwelling does not exceed the price
    lam, capped = L.lambda_from_inputs([v], [d], [p], f)
    assert 0.0 <= lam[0] <= 1.0
    assert not capped[0]
    assert math.isclose(lam[0], ratio, rel_tol=1e-12, abs_tol=0.0)


@given(v=land_values, d=densities, p=prices, f=factors)
def test_lambda_always_reported_on_unit_interval_and_capping_is_flagged(v, d, p, f):
    lam, capped = L.lambda_from_inputs([v], [d], [p], f)
    ratio = v / (f * d) / p
    assert 0.0 <= lam[0] <= 1.0
    assert bool(capped[0]) == (ratio > 1.0)
    if capped[0]:
        assert lam[0] == 1.0


def test_missing_land_value_passes_through_as_nan_and_is_not_capped():
    lam, capped = L.lambda_from_inputs([np.nan, 10.0], [30.0, 30.0], [1e5, 1e5])
    assert np.isnan(lam[0]) and not capped[0]
    assert math.isclose(lam[1], 10.0 / 30.0 / 1e5)


@pytest.mark.parametrize(
    "v, d, p, f",
    [(-1.0, 30.0, 1e5, 1.0), (1.0, 0.0, 1e5, 1.0), (1.0, 30.0, 0.0, 1.0), (1.0, 30.0, 1e5, 0.0)],
)
def test_invalid_inputs_are_rejected(v, d, p, f):
    with pytest.raises(ValueError):
        L.lambda_from_inputs([v], [d], [p], f)


# --------------------------------------------------------------------------
# Dwelling-weighted aggregation is a convex combination
# --------------------------------------------------------------------------

lams = st.one_of(st.floats(min_value=0.0, max_value=1.0), st.just(float("nan")))
# Weights are 0 or at least 1e-3 so rescaling by k >= 1e-6 cannot underflow to 0.
weights = st.one_of(st.just(0.0), st.floats(min_value=1e-3, max_value=1e7))


@st.composite
def lambda_weight_pairs(draw):
    n = draw(st.integers(min_value=1, max_value=40))
    xs = draw(st.lists(lams, min_size=n, max_size=n))
    ws = draw(st.lists(weights, min_size=n, max_size=n))
    keep = [i for i, x in enumerate(xs) if not math.isnan(x)]
    assume(keep and sum(ws[i] for i in keep) > 0)
    return np.array(xs), np.array(ws)


@given(pair=lambda_weight_pairs())
def test_weighted_mean_is_a_convex_combination(pair):
    xs, ws = pair
    keep = ~np.isnan(xs)
    m = L.weighted_mean(xs, ws)
    alpha = L.convex_weights(ws[keep])
    assert np.all(alpha >= 0)
    assert math.isclose(alpha.sum(), 1.0, rel_tol=1e-12)
    assert math.isclose(m, float(np.dot(alpha, xs[keep])), rel_tol=1e-12, abs_tol=1e-15)
    # Bounded by the extremes of the values it averages (tolerance: fp rounding).
    assert xs[keep].min() - 1e-12 <= m <= xs[keep].max() + 1e-12


@given(pair=lambda_weight_pairs(), k=st.floats(min_value=1e-6, max_value=1e6))
def test_weighted_mean_scale_invariant(pair, k):
    xs, ws = pair
    assert math.isclose(L.weighted_mean(xs, ws), L.weighted_mean(xs, ws * k), rel_tol=1e-9, abs_tol=1e-12)


@given(pair=lambda_weight_pairs(), data=st.data())
def test_weighted_mean_permutation_invariant(pair, data):
    xs, ws = pair
    perm = data.draw(st.permutations(range(len(xs))))
    perm = np.array(perm)
    assert math.isclose(L.weighted_mean(xs, ws), L.weighted_mean(xs[perm], ws[perm]), rel_tol=1e-9, abs_tol=1e-12)


@given(x=st.floats(min_value=0.0, max_value=1.0), ws=st.lists(st.floats(min_value=1e-3, max_value=1e6), min_size=1, max_size=20))
def test_weighted_mean_of_constant_is_that_constant(x, ws):
    assert math.isclose(L.weighted_mean([x] * len(ws), ws), x, rel_tol=1e-12, abs_tol=1e-15)


def test_weighted_mean_rejects_negative_or_zero_total_weights():
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [1.0, -1.0])
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [0.0, 0.0])


# --------------------------------------------------------------------------
# Density sensitivity moves lambda in the expected direction
# --------------------------------------------------------------------------

@given(v=land_values, d=densities, p=prices, f1=factors, f2=factors)
def test_lambda_non_increasing_in_density_factor(v, d, p, f1, f2):
    lo, hi = sorted((f1, f2))
    lam_lo, _ = L.lambda_from_inputs([v], [d], [p], lo)
    lam_hi, _ = L.lambda_from_inputs([v], [d], [p], hi)
    # More homes per hectare -> less land per home -> lambda no higher.
    assert lam_hi[0] <= lam_lo[0] * (1 + 1e-12)


def _variant(name):
    return next(v for v in L.VARIANTS if v.name == name)


@given(v=land_values, d=densities, p=prices)
def test_density_variants_bracket_central_with_exact_ratios(v, d, p):
    central, _ = L.lambda_from_inputs([v], [d], [p], _variant("central").density_factor)
    plus, plus_capped = L.lambda_from_inputs([v], [d], [p], _variant("density_plus25").density_factor)
    minus, _ = L.lambda_from_inputs([v], [d], [p], _variant("density_minus25").density_factor)
    assert minus[0] <= central[0] * (1 + 1e-12)
    assert central[0] <= plus[0] * (1 + 1e-12)
    if not plus_capped[0]:
        assert math.isclose(plus[0], 1.25 * central[0], rel_tol=1e-12, abs_tol=1e-300)
    raw_central = v / d / p
    if raw_central <= 1.0:
        assert math.isclose(minus[0], 0.75 * central[0], rel_tol=1e-12, abs_tol=1e-300)


# --------------------------------------------------------------------------
# The table builder keeps each authority exactly once (generated universes)
# --------------------------------------------------------------------------

SHEETS = ("floor_adjusted", "unadjusted")


def _synthetic_inputs(rows):
    """Build mhclg/hpi/ctsop frames from (code, mhclg_code, has_mhclg, has_hpi, v, d, p, n)."""
    mh, hp, ct = [], [], []
    for code, mcode, has_mhclg, has_hpi, v, d, p, n in rows:
        ct.append({"la_code": code, "la_name": code, "region_code": "E12000009",
                   "region_name": "South West", "all_properties": n})
        if has_hpi:
            hp.append({"la_code": code, "la_name": code, "date": L.HPI_DATE, "average_price": p})
        if has_mhclg:
            for sheet in SHEETS:
                for scen, dd in (("low", d), ("medium", d), ("high", d)):
                    for pct in (25, 40, 50, 60, 75):
                        mh.append({"la_code_mhclg": mcode, "la_name": code, "region": "South West",
                                   "sheet": sheet, "density_scenario": scen, "dwellings_per_ha": dd,
                                   "price_percentile": pct, "land_value_gbp_per_ha": v * (pct / 50)})
    return pd.DataFrame(mh), pd.DataFrame(hp), pd.DataFrame(ct)


@st.composite
def synthetic_universe(draw):
    n = draw(st.integers(min_value=1, max_value=8))
    nums = draw(st.lists(st.integers(min_value=100, max_value=999), min_size=n, max_size=n, unique=True))
    rows = []
    for k in nums:
        rows.append((f"E07000{k}", f"E07000{k}", True, True,
                     draw(st.floats(min_value=0, max_value=5e7)),
                     draw(st.floats(min_value=30, max_value=400)),
                     draw(st.floats(min_value=5e4, max_value=2e6)),
                     draw(st.integers(min_value=1, max_value=500_000))))
    if draw(st.booleans()):  # an authority published by MHCLG under its pre-2023 code
        old, new = next(iter(MHCLG_TO_2023.items()))
        rows.append((new, old, True, True, 1e6, 40.0, 2e5, 1000))
    if draw(st.booleans()):  # Isles of Scilly with no MHCLG or HPI row, donor present
        rows.append((CORNWALL, CORNWALL, True, True, 2e6, 30.0, 3e5, 5000))
        rows.append((ISLES_OF_SCILLY, ISLES_OF_SCILLY, False, False, 0, 30.0, 1.0, 1000))
    return rows


@settings(deadline=None, max_examples=60)
@given(rows=synthetic_universe())
def test_builder_keeps_every_authority_once_with_lambda_on_unit_interval(rows):
    mh, hp, ct = _synthetic_inputs(rows)
    out = L.build_la_table(mh, hp, ct, expected_n=None)
    assert sorted(out["la_code"]) == sorted(ct["la_code"])
    assert not out["la_code"].duplicated().any()
    lam_cols = [c for c in out.columns if c.startswith("lambda_raw")]
    vals = out[lam_cols].to_numpy(dtype=float)
    assert np.all((vals[~np.isnan(vals)] >= 0) & (vals[~np.isnan(vals)] <= 1))
    assert out["lambda_raw"].notna().all()
    assert (out["lambda_raw_density_plus25"] >= out["lambda_raw"] - 1e-6).all()
    assert (out["lambda_raw_density_minus25"] <= out["lambda_raw"] + 1e-6).all()
    if ISLES_OF_SCILLY in set(out["la_code"]):
        s = out.set_index("la_code")
        assert s.loc[ISLES_OF_SCILLY, "method_flag"] == f"imputed_from_{CORNWALL}"
        assert s.loc[ISLES_OF_SCILLY, "lambda_raw"] == s.loc[CORNWALL, "lambda_raw"]
    for old, new in MHCLG_TO_2023.items():
        assert old not in set(out["la_code"])
        if new in set(out["la_code"]):
            assert out.set_index("la_code").loc[new, "method_flag"] == "computed_code_crosswalk"


def test_builder_rejects_missing_authority_without_donor():
    mh, hp, ct = _synthetic_inputs([("E07000101", "E07000101", True, True, 1e6, 40.0, 2e5, 10)])
    ct = pd.concat([ct, pd.DataFrame([{"la_code": "E07000102", "la_name": "x", "region_code": "E12000009",
                                       "region_name": "South West", "all_properties": 5}])])
    with pytest.raises(ValueError, match="no central lambda"):
        L.build_la_table(mh, hp, ct, expected_n=None)


def test_builder_rejects_duplicate_or_unknown_codes():
    mh, hp, ct = _synthetic_inputs([("E07000101", "E07000101", True, True, 1e6, 40.0, 2e5, 10)])
    with pytest.raises(ValueError, match="duplicated"):
        L.build_la_table(mh, hp, pd.concat([ct, ct]), expected_n=None)
    hp2 = pd.concat([hp, hp.assign(la_code="E07000999")])
    with pytest.raises(ValueError, match="outside the universe"):
        L.build_la_table(mh, hp2, ct, expected_n=None)
    with pytest.raises(ValueError, match="expected 296"):
        L.build_la_table(mh, hp, ct)
