"""Properties of the relative (float32-spacing) closed-form tolerance.

``pipeline_direct.validate`` refuses to write results unless every household's
full-model change in net income equals ``CT - r * L`` to within
``FLOAT32_ULP_TOLERANCE`` float32 spacings of the household's largest total.
These tests pin down what that guard accepts and rejects on synthetic data.
"""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from uk_lvt.validation import (
    FLOAT32_ULP_TOLERANCE,
    closed_form_gap_check,
    float32_ulp,
    gap_tolerance,
    household_scale,
)

# Household totals are pounds a year: from zero to well above the largest
# incomes in the survey (float32 spacing reaches £1 at 2**23).
money = st.floats(min_value=0.0, max_value=5e8, allow_nan=False, allow_infinity=False)
positive_money = st.floats(min_value=1.0, max_value=5e8)

# Timing is not what these properties test. On a busy shared host the
# per-example deadline and the slow-generation health check both fail on
# the array tests below, so the array tests turn both off and the scalar
# tests turn off the deadline; the example counts are unchanged.
NO_DEADLINE = settings(deadline=None)
SLOW_GENERATION = settings(NO_DEADLINE, suppress_health_check=[HealthCheck.too_slow])


def test_float32_ulp_matches_known_spacings():
    assert float32_ulp(2.0**23) == 1.0
    assert float32_ulp(9.3e6) == 1.0
    assert float32_ulp(2.0**21) == 0.25
    assert float32_ulp(1_000_000.0) == 0.0625
    assert float32_ulp(-1_000_000.0) == 0.0625


@NO_DEADLINE
@given(positive_money)
def test_tolerance_is_eight_float32_spacings(scale):
    assert gap_tolerance(scale) == FLOAT32_ULP_TOLERANCE * float32_ulp(scale)
    assert FLOAT32_ULP_TOLERANCE == 8


@NO_DEADLINE
@given(money, money)
def test_tolerance_is_monotone_in_scale(a, b):
    lo, hi = sorted((a, b))
    assert gap_tolerance(lo) <= gap_tolerance(hi)


@NO_DEADLINE
@given(
    st.floats(min_value=1e-3, max_value=1e6), st.integers(min_value=-10, max_value=10)
)
def test_tolerance_scales_with_powers_of_two(scale, k):
    assert gap_tolerance(scale * 2.0**k) == 2.0**k * gap_tolerance(scale)


@NO_DEADLINE
@given(st.floats(min_value=0.0, max_value=2_000_000.0))
def test_no_looser_than_one_pound_below_two_million(scale):
    # The previous guard allowed |gap| <= £1 for every household. For any
    # household whose largest total is under £2m the new guard is at least as
    # strict, so it cannot hide an interaction the old guard would have caught.
    assert gap_tolerance(scale) <= 1.0


@settings(SLOW_GENERATION, max_examples=200)
@given(
    arrays(np.float64, 20, elements=positive_money),
    st.integers(min_value=0, max_value=19),
    st.integers(min_value=0, max_value=FLOAT32_ULP_TOLERANCE),
)
def test_gaps_within_tolerance_pass(scale, i, k):
    arith = np.zeros_like(scale)
    model = arith.copy()
    model[i] = k * float32_ulp(scale[i])
    result = closed_form_gap_check(model, arith, scale, np.ones_like(scale))
    assert result["passed"]
    assert result["share_households_with_benefit_interaction"] == 0.0
    assert result["max_delta_gap_float32_ulps"] <= FLOAT32_ULP_TOLERANCE


@settings(SLOW_GENERATION, max_examples=200)
@given(
    arrays(np.float64, 20, elements=positive_money),
    st.integers(min_value=0, max_value=19),
    st.floats(min_value=1.01, max_value=1e3),
)
def test_gaps_beyond_tolerance_fail(scale, i, factor):
    arith = np.zeros_like(scale)
    model = arith.copy()
    model[i] = factor * gap_tolerance(scale[i])
    weights = np.ones_like(scale)
    result = closed_form_gap_check(model, arith, scale, weights)
    assert not result["passed"]
    assert result["share_households_with_benefit_interaction"] == 100 / len(scale)


@SLOW_GENERATION
@given(arrays(np.float64, 30, elements=positive_money))
def test_one_pound_interaction_caught_for_ordinary_households(scale):
    # A real benefit interaction moves net income by pounds. For households
    # whose totals are under £2m a £1.01 gap always fails the guard.
    scale = np.minimum(scale, 2_000_000.0)
    arith = np.zeros_like(scale)
    model = np.full_like(scale, 1.01)
    result = closed_form_gap_check(model, arith, scale, np.ones_like(scale))
    assert not result["passed"]
    assert result["share_households_with_benefit_interaction"] == 100.0
    assert result["share_households_gap_over_threshold_gbp"] == 100.0


def _f32(x):
    return np.float32(x)


@settings(NO_DEADLINE, max_examples=500)
@given(
    market=money,
    benefits=money,
    other_tax=money,
    council_tax=st.floats(min_value=0.0, max_value=5e4),
    ctr_share=st.floats(min_value=0.0, max_value=1.0),
    land=st.floats(min_value=0.0, max_value=5e8),
    rate=st.sampled_from([0.0, 0.005, 0.0078667, 0.01, 0.02, 0.05]),
)
def test_float32_swap_rounding_stays_within_tolerance(
    market, benefits, other_tax, council_tax, ctr_share, land, rate
):
    """A float32 model of the swap passes the guard: rounding alone never trips it.

    Baseline net income = market + (benefits + CTR) - (other tax + council
    tax); reform net income = market + benefits - (other tax + rate * land).
    Every sum is rounded to float32, as the model's household totals are. The
    closed form uses the same float32 inputs in float64 arithmetic, exactly as
    ``pipeline_direct`` reads them.
    """
    m, b, t = _f32(market), _f32(benefits), _f32(other_tax)
    ct = _f32(council_tax)
    ctr = _f32(council_tax * ctr_share)
    land32 = _f32(land)

    tax_b = _f32(t + ct)
    ben_b = _f32(b + ctr)
    net_b = _f32(_f32(m + ben_b) - tax_b)

    lvt = _f32(_f32(rate) * land32)
    tax_r = _f32(t + lvt)
    ben_r = b
    net_r = _f32(_f32(m + ben_r) - tax_r)

    model_delta = np.array([float(net_r) - float(net_b)])
    arith_delta = np.array([(float(ct) - float(ctr)) - rate * float(land32)])
    scale = household_scale(
        [net_b], [m], [ben_b], [tax_b], [net_r], [m], [ben_r], [tax_r]
    )
    result = closed_form_gap_check(model_delta, arith_delta, scale, np.ones(1))
    assert result["passed"], result
