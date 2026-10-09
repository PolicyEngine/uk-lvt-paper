"""Property-based tests for the raw land-share arithmetic in uk_lvt.land.lambda_raw."""

import math
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from uk_lvt.land import lambda_raw as L
from uk_lvt.land.geography import CORNWALL, ISLES_OF_SCILLY, MHCLG_TO_2023

# Ranges cover the data with margin: MHCLG values reach about 1.9e8 GBP/ha,
# densities are 30-400 dph, UK HPI authority prices are about 1.1e5-1.5e6.
land_values = st.floats(min_value=0.0, max_value=1e9, allow_nan=False)
densities = st.floats(min_value=1.0, max_value=1_000.0, allow_nan=False)
prices = st.floats(min_value=1e3, max_value=1e8, allow_nan=False)
factors = st.floats(min_value=0.05, max_value=20.0, allow_nan=False)

# The properties here are about arithmetic, not speed. Hypothesis's timing
# checks (the 200 ms deadline and the slow-generation health check) fail on a
# loaded shared host, so every property test in this module switches them off.
TIMING_FREE = settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])


# --------------------------------------------------------------------------
# 0 <= lambda <= 1
# --------------------------------------------------------------------------

@TIMING_FREE
@given(v=land_values, d=densities, p=prices, f=factors)
def test_lambda_in_unit_interval_and_exact_for_valid_inputs(v, d, p, f):
    ratio = v / (f * d) / p
    assume(ratio <= 1.0)  # valid: plot land per dwelling does not exceed the price
    lam, capped = L.lambda_from_inputs([v], [d], [p], f)
    assert 0.0 <= lam[0] <= 1.0
    assert not capped[0]
    assert math.isclose(lam[0], ratio, rel_tol=1e-12, abs_tol=0.0)


@TIMING_FREE
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
# Any finite non-negative float, subnormals and the largest float included:
# two of the largest already sum to infinity.
any_finite_weights = st.one_of(
    weights,
    st.floats(min_value=0.0, max_value=sys.float_info.max, allow_nan=False, allow_infinity=False),
    st.just(sys.float_info.max),
)


@st.composite
def lambda_weight_pairs(draw, weight_strategy=weights):
    n = draw(st.integers(min_value=1, max_value=40))
    xs = draw(st.lists(lams, min_size=n, max_size=n))
    ws = draw(st.lists(weight_strategy, min_size=n, max_size=n))
    assume(any(ws[i] > 0 for i, x in enumerate(xs) if not math.isnan(x)))
    return np.array(xs), np.array(ws)


@TIMING_FREE
@given(pair=lambda_weight_pairs(any_finite_weights))
def test_weighted_mean_is_a_convex_combination(pair):
    xs, ws = pair
    keep = ~np.isnan(xs)
    m = L.weighted_mean(xs, ws)
    alpha = L.convex_weights(ws[keep])
    assert np.all(np.isfinite(alpha)) and np.all(alpha >= 0)
    assert math.isclose(alpha.sum(), 1.0, rel_tol=1e-12)
    assert math.isclose(m, float(np.dot(alpha, xs[keep])), rel_tol=1e-12, abs_tol=1e-15)
    # Bounded by the extremes of the values it averages (tolerance: fp rounding).
    assert xs[keep].min() - 1e-12 <= m <= xs[keep].max() + 1e-12


@TIMING_FREE
@given(pair=lambda_weight_pairs(any_finite_weights))
def test_weighted_mean_matches_exact_rational_mean(pair):
    """Differential check against exact arithmetic: sum(w x) / sum(w) in rationals."""
    xs, ws = pair
    keep = ~np.isnan(xs)
    w = [Fraction(v) for v in ws[keep]]
    x = [Fraction(v) for v in xs[keep]]
    exact = sum(wi * xi for wi, xi in zip(w, x)) / sum(w)
    # Every term is non-negative, so rounding error is relative (about 2n
    # machine epsilons); the absolute floor covers subnormal underflow.
    assert math.isclose(L.weighted_mean(xs, ws), float(exact), rel_tol=1e-12, abs_tol=1e-300)


def test_weighted_mean_with_weights_whose_sum_overflows():
    big = sys.float_info.max
    assert math.isinf(sum([big, big]))
    for w in ([1e308, 1e308], [big, big], [big, big, big]):
        assert math.isclose(L.weighted_mean([0.2, 0.4, 0.6][: len(w)], w),
                            float(np.mean([0.2, 0.4, 0.6][: len(w)])), rel_tol=1e-12)
    assert math.isclose(L.weighted_mean([0.2, 0.4], [big, 1.0]), 0.2, rel_tol=1e-12)
    alpha = L.convex_weights([big, big, 0.0])
    np.testing.assert_allclose(alpha, [0.5, 0.5, 0.0], rtol=1e-15)


@TIMING_FREE
@given(pair=lambda_weight_pairs(), k=st.floats(min_value=1e-6, max_value=1e6))
def test_weighted_mean_scale_invariant(pair, k):
    xs, ws = pair
    assert math.isclose(L.weighted_mean(xs, ws), L.weighted_mean(xs, ws * k), rel_tol=1e-9, abs_tol=1e-12)


@TIMING_FREE
@given(pair=lambda_weight_pairs(any_finite_weights), data=st.data())
def test_weighted_mean_permutation_invariant(pair, data):
    xs, ws = pair
    perm = data.draw(st.permutations(range(len(xs))))
    perm = np.array(perm)
    assert math.isclose(L.weighted_mean(xs, ws), L.weighted_mean(xs[perm], ws[perm]), rel_tol=1e-9, abs_tol=1e-12)


@TIMING_FREE
@given(x=st.floats(min_value=0.0, max_value=1.0), ws=st.lists(any_finite_weights, min_size=1, max_size=20))
def test_weighted_mean_of_constant_is_that_constant(x, ws):
    assume(any(w > 0 for w in ws))
    assert math.isclose(L.weighted_mean([x] * len(ws), ws), x, rel_tol=1e-12, abs_tol=1e-15)


def test_weighted_mean_rejects_negative_zero_or_non_finite_weights():
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [1.0, -1.0])
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [0.0, 0.0])
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [1.0, float("inf")])
    with pytest.raises(ValueError):
        L.weighted_mean([0.2, 0.4], [1.0, float("nan")])
    with pytest.raises(ValueError):
        L.convex_weights([])


# --------------------------------------------------------------------------
# Density sensitivity moves lambda in the expected direction
# --------------------------------------------------------------------------

@TIMING_FREE
@given(v=land_values, d=densities, p=prices, f1=factors, f2=factors)
def test_lambda_non_increasing_in_density_factor(v, d, p, f1, f2):
    lo, hi = sorted((f1, f2))
    lam_lo, _ = L.lambda_from_inputs([v], [d], [p], lo)
    lam_hi, _ = L.lambda_from_inputs([v], [d], [p], hi)
    # More homes per hectare -> less land per home -> lambda no higher.
    assert lam_hi[0] <= lam_lo[0] * (1 + 1e-12)


def _variant(name):
    return next(v for v in L.VARIANTS if v.name == name)


# Density factor f of each variant: the density used is f x the published one.
EXPECTED_DENSITY_FACTORS = {
    "central": 1.0,
    "density_minus25": 0.75,      # density -25%: lambda x 4/3
    "density_plus25": 1.25,       # density +25%: lambda x 0.8
    "gross_net_lower": 4.0 / 3.0,  # gross/net-density bound, lambda -25%
    "gross_net_upper": 0.8,       # gross/net-density bound, lambda +25%
    "price_p25": 1.0,
    "price_p75": 1.0,
    "unadjusted": 1.0,
}
DENSITY_VARIANTS = ("density_minus25", "density_plus25", "gross_net_lower", "gross_net_upper")


def test_variant_density_factors_are_as_documented():
    assert len(set(L.VARIANT_NAMES)) == len(L.VARIANT_NAMES)
    assert {v.name: v.density_factor for v in L.VARIANTS} == EXPECTED_DENSITY_FACTORS
    central = _variant("central")
    for name in DENSITY_VARIANTS:  # only the density changes
        v = _variant(name)
        assert (v.sheet, v.density_scenario, v.price_percentile) == (
            central.sheet, central.density_scenario, central.price_percentile)


@TIMING_FREE
@given(v=land_values, d=densities, p=prices, f=factors)
def test_density_factor_multiplies_the_published_density(v, d, p, f):
    """f x d homes per hectare: the same as publishing a density of f x d."""
    lam, capped = L.lambda_from_inputs([v], [d], [p], f)
    literal, literal_capped = L.lambda_from_inputs([v], [f * d], [p], 1.0)
    assert math.isclose(lam[0], literal[0], rel_tol=1e-12, abs_tol=1e-300)
    assert capped[0] == literal_capped[0]


@TIMING_FREE
@given(v=land_values, d=densities, p=prices)
def test_density_variants_scale_central_lambda_exactly(v, d, p):
    lam = {name: L.lambda_from_inputs([v], [d], [p], _variant(name).density_factor)[0][0]
           for name in ("central",) + DENSITY_VARIANTS}
    # Density ±25%: more homes per hectare, less land per home, lower lambda.
    assert lam["density_plus25"] <= lam["central"] * (1 + 1e-12)
    assert lam["central"] <= lam["density_minus25"] * (1 + 1e-12)
    # Gross/net-density bound: lambda ±25%.
    assert lam["gross_net_lower"] <= lam["central"] * (1 + 1e-12)
    assert lam["central"] <= lam["gross_net_upper"] * (1 + 1e-12)
    raw_central = v / d / p
    # Exact ratios to central wherever neither side is capped at 1.
    ratios = {"density_minus25": 4.0 / 3.0, "density_plus25": 0.8,
              "gross_net_lower": 0.75, "gross_net_upper": 1.25}
    for name, ratio in ratios.items():
        if raw_central <= 1.0 and raw_central * ratio <= 1.0:
            assert math.isclose(lam[name], ratio * lam["central"], rel_tol=1e-12, abs_tol=1e-300), name


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


@settings(parent=TIMING_FREE, max_examples=60)
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
    assert (out["lambda_raw_density_plus25"] <= out["lambda_raw"] + 1e-6).all()
    assert (out["lambda_raw_density_minus25"] >= out["lambda_raw"] - 1e-6).all()
    assert (out["lambda_raw_gross_net_upper"] >= out["lambda_raw"] - 1e-6).all()
    assert (out["lambda_raw_gross_net_lower"] <= out["lambda_raw"] + 1e-6).all()
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
