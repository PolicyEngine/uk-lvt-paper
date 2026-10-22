"""Properties of the budget-neutral rate solver used by ``pipeline_direct``."""

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from uk_lvt.validation import budget_neutral_rate

N = st.integers(min_value=1, max_value=200)
weight = st.floats(min_value=1e-3, max_value=1e5)
# Land and replaced revenue per household are zero or at least a penny.
land = st.one_of(st.just(0.0), st.floats(min_value=0.01, max_value=1e8))
revenue = st.one_of(st.just(0.0), st.floats(min_value=0.01, max_value=1e5))

# Timing is not what these properties test. On a busy shared host the
# per-example deadline and the slow-generation health check both fail on
# arrays like these, so both are off; the example counts are unchanged.
NO_DEADLINE = settings(deadline=None)
SLOW_GENERATION = settings(NO_DEADLINE, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def households(draw):
    n = draw(N)
    w = draw(arrays(np.float64, n, elements=weight))
    L = draw(arrays(np.float64, n, elements=land))
    R = draw(arrays(np.float64, n, elements=revenue))
    assume(np.sum(w * L) > 0)
    return w, L, R


@settings(SLOW_GENERATION, max_examples=300)
@given(households())
def test_solved_rate_is_revenue_neutral(data):
    # P1: |sum w (r L - R)| <= 1e-9 * sum w R.
    w, L, R = data
    rate, _, _ = budget_neutral_rate(R, L, w)
    gap = abs(float(np.sum(w * (rate * L - R))))
    assert gap <= 1e-9 * float(np.sum(w * R))


@settings(SLOW_GENERATION, max_examples=300)
@given(households())
def test_rate_is_non_negative_and_reports_totals(data):
    w, L, R = data
    rate, revenue_bn, land_bn = budget_neutral_rate(R, L, w)
    assert rate >= 0
    assert revenue_bn == float(np.sum(R * w)) / 1e9
    assert land_bn == float(np.sum(L * w)) / 1e9


@settings(SLOW_GENERATION, max_examples=200)
@given(households(), st.floats(min_value=1e-3, max_value=1e3))
def test_rate_is_invariant_to_scaling_weights(data, k):
    # P7: multiplying every weight by k > 0 leaves the rate unchanged.
    w, L, R = data
    r1, _, _ = budget_neutral_rate(R, L, w)
    r2, _, _ = budget_neutral_rate(R, L, k * w)
    assert r2 == pytest.approx(r1, rel=1e-12, abs=1e-300)


@settings(SLOW_GENERATION, max_examples=200)
@given(households(), st.floats(min_value=1.01, max_value=10.0))
def test_rate_rises_with_revenue_target(data, k):
    w, L, R = data
    assume(np.sum(w * R) > 0)
    r1, _, _ = budget_neutral_rate(R, L, w)
    r2, _, _ = budget_neutral_rate(k * R, L, w)
    assert r2 > r1


def test_matches_the_pipeline_arithmetic_bit_for_bit():
    # The solver must perform exactly the operations pipeline_direct used
    # before it was factored out, so that no published digit can move.
    rng = np.random.default_rng(20261005)
    w = rng.uniform(100, 5000, 5000)
    L = rng.lognormal(12, 1.5, 5000)
    R = rng.uniform(0, 4000, 5000)
    expected_rate = (float(np.sum(R * w)) / 1e9) / (float(np.sum(L * w)) / 1e9)
    rate, _, _ = budget_neutral_rate(R, L, w)
    assert rate == expected_rate


def test_zero_land_base_raises():
    with pytest.raises(ValueError):
        budget_neutral_rate(np.ones(3), np.zeros(3), np.ones(3))
