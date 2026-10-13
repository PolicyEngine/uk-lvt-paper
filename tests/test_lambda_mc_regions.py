"""``analysis/lambda_mc.py`` maps regions to land shares under pandas 3.

Under pandas 3 a string column's ``.values`` is a pandas string array, which
``np.char.find`` rejects; the script raised before writing
``results/lambda_mc.json``. These tests run the mapping on such a column.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import lambda_mc  # noqa: E402


@pytest.mark.parametrize("dtype", ["str", "string", object])
def test_region_lambda_maps_every_region(dtype):
    regions = list(lambda_mc.LAMBDA)
    df = pd.DataFrame({"region": pd.Series(regions, dtype=dtype)})
    lam = lambda_mc.region_lambda(df)
    assert lam.tolist() == [lambda_mc.LAMBDA[r] for r in regions]
    assert np.all((lam > 0) & (lam < 1))


def test_region_lambda_rejects_unknown_region():
    df = pd.DataFrame({"region": pd.Series(["LONDON", "ATLANTIS"], dtype="str")})
    with pytest.raises(SystemExit):
        lambda_mc.region_lambda(df)
