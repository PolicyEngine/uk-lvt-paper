"""Rate solver and the closed-form validation tolerance.

Two pieces of arithmetic that ``pipeline_direct`` relies on, kept free of any
policyengine import so they can be property-tested on synthetic arrays.

Budget-neutral rate
    ``r = sum(w * R) / sum(w * L)`` for replaced revenue ``R`` and land ``L``
    per household. ``budget_neutral_rate`` performs exactly the floating-point
    operations ``pipeline_direct`` has always used (both sums scaled to
    billions before dividing), so routing the pipeline through it leaves every
    published digit unchanged.

Closed-form tolerance
    The model computes household totals in float32. A household's change in
    ``household_net_income`` is the difference of two float32 sums, so it
    carries rounding error that scales with the size of the household's
    largest total, not a fixed number of pounds: between £8.4m and £16.8m
    (2**23 to 2**24) the float32 spacing is £1.00. The check therefore allows
    a gap of at most ``FLOAT32_ULP_TOLERANCE`` float32 spacings (ulps) of the
    largest absolute value among the household's net income, market income,
    benefits and tax, in either run.
    For any household whose largest total is below £2 million this is no
    looser than the previous absolute £1 tolerance (8 spacings of a number
    below 2**21 is at most £1); above that it grows with the float32 spacing.
"""

from __future__ import annotations

import numpy as np

# Allowed closed-form gap, in float32 spacings of the household's largest
# total. The largest gap measured on the paper's pinned inputs is recorded in
# results/lvt_results.json (validation.checks[].max_delta_gap_float32_ulps);
# on data release 1.57.4 with policyengine-uk 2.93.0 it was at most about 4.
# 8 leaves headroom while still catching a real benefit interaction, which
# moves net income by pounds rather than by float32 spacings.
FLOAT32_ULP_TOLERANCE = 8

# Reporting threshold (not a guard): the paper states the share of households
# whose closed-form gap exceeds £1.
REPORTING_THRESHOLD_GBP = 1.0


def budget_neutral_rate(
    revenue: np.ndarray, land: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    """Flat land tax rate that raises the weighted ``revenue`` total.

    Returns ``(rate, revenue_bn, land_bn)``. Raises ``ValueError`` when the
    weighted land base is not positive, because no finite rate then exists.
    """
    revenue_bn = float(np.sum(revenue * weights)) / 1e9
    land_bn = float(np.sum(land * weights)) / 1e9
    if not land_bn > 0:
        raise ValueError(f"Weighted land base must be positive, got £{land_bn}bn")
    return revenue_bn / land_bn, revenue_bn, land_bn


def float32_ulp(x) -> np.ndarray:
    """Spacing between adjacent float32 numbers at ``|x|``, as float64."""
    magnitude = np.abs(np.asarray(x, dtype=np.float64)).astype(np.float32)
    return np.spacing(magnitude).astype(np.float64)


def household_scale(*totals) -> np.ndarray:
    """Largest absolute value across household-level arrays, per household."""
    stacked = np.vstack([np.abs(np.asarray(t, dtype=np.float64)) for t in totals])
    return stacked.max(axis=0)


def gap_tolerance(scale, ulps: float = FLOAT32_ULP_TOLERANCE) -> np.ndarray:
    """Allowed absolute gap for households whose largest total is ``scale``."""
    return ulps * float32_ulp(scale)


def closed_form_gap_check(
    model_delta: np.ndarray,
    arithmetic_delta: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
    ulps: float = FLOAT32_ULP_TOLERANCE,
    reporting_threshold_gbp: float = REPORTING_THRESHOLD_GBP,
) -> dict:
    """Compare full-model and closed-form household changes.

    Returns the largest gap in pounds and in float32 spacings, the weighted
    share of households whose gap exceeds the tolerance (a gap that float32
    rounding cannot explain, i.e. a benefit interaction), the weighted share
    whose gap exceeds the £ reporting threshold, and ``passed``.
    """
    gap = np.abs(np.asarray(model_delta, dtype=np.float64) - arithmetic_delta)
    ulp = float32_ulp(scale)
    gap_in_ulps = np.divide(gap, ulp, out=np.zeros_like(gap), where=gap > 0)
    exceeds = gap > ulps * ulp
    total_weight = np.sum(weights)
    return {
        "max_abs_delta_gap": float(np.max(gap)) if gap.size else 0.0,
        "max_delta_gap_float32_ulps": float(np.max(gap_in_ulps)) if gap.size else 0.0,
        "share_households_with_benefit_interaction": 100
        * float(np.sum(exceeds * weights) / total_weight),
        "share_households_gap_over_threshold_gbp": 100
        * float(np.sum((gap > reporting_threshold_gbp) * weights) / total_weight),
        "passed": not bool(np.any(exceeds)),
    }
