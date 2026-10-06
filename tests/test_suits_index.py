"""The Suits index in ``analysis/deep_dive.py`` runs on the locked NumPy.

``deep_dive.py`` used ``np.trapz``, which NumPy 2.4 removed, so the script
raised ``AttributeError`` before writing ``results/deep_dive.json``. It now
uses ``np.trapezoid``; these tests exercise that code path on synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import deep_dive  # noqa: E402


def _reference_suits(tax, y, w):
    """Suits index with the trapezoidal rule written out by hand."""
    order = np.argsort(y, kind="stable")
    t = (tax * w)[order]
    inc = (y * w)[order]
    x = np.concatenate([[0.0], np.cumsum(inc) / inc.sum()])
    z = np.concatenate([[0.0], np.cumsum(t) / t.sum()])
    area = np.sum(np.diff(x) * (z[1:] + z[:-1]) / 2.0)
    return 1 - 2 * area


positive = st.floats(min_value=1.0, max_value=1e6)


@settings(max_examples=200)
@given(
    arrays(np.float64, 50, elements=positive),
    arrays(np.float64, 50, elements=positive),
    arrays(np.float64, 50, elements=st.floats(min_value=0.1, max_value=1e4)),
)
def test_suits_matches_hand_trapezoid(tax, y, w):
    assert deep_dive.suits_index(tax, y, w) == pytest.approx(
        _reference_suits(tax, y, w), abs=1e-12
    )


@settings(max_examples=200)
@given(
    arrays(np.float64, 50, elements=positive),
    arrays(np.float64, 50, elements=st.floats(min_value=0.1, max_value=1e4)),
    st.floats(min_value=1e-4, max_value=10.0),
)
def test_proportional_tax_has_zero_suits_index(y, w, rate):
    assert deep_dive.suits_index(rate * y, y, w) == pytest.approx(0.0, abs=1e-9)


def test_suits_bounds_on_extreme_taxes():
    y = np.arange(1.0, 101.0)
    w = np.ones(100)
    poll = deep_dive.suits_index(np.ones(100), y, w)  # equal per head: regressive
    top_only = deep_dive.suits_index((y == 100).astype(float), y, w)
    assert -1 < poll < 0
    assert 0 < top_only <= 1
