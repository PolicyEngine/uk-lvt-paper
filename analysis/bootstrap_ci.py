"""Bootstrap confidence intervals for the paper's headline statistics.

Survey bootstrap over household weights: each replicate multiplies every
household weight by an independent Poisson(1) draw (Rao-Wu style weight
bootstrap), re-solves the budget-neutral rate on the resampled population,
and recomputes the headline statistics. The household-level delta arithmetic
(delta = council tax saved - rate x land) is exact, so no re-simulation is
needed; PolicyEngine runs once to extract the vectors.

Writes ``results/bootstrap_ci.json`` with the point estimate, bootstrap
standard error and 95 per cent percentile interval for each statistic.

Run:  python analysis/bootstrap_ci.py            (B=1000, ~10-20 min)
      python analysis/bootstrap_ci.py --reps 200 (quick check)

Caveat recorded in the output: this captures sampling variation in the
survey weights only, not the imputation variance of the land values
themselves (WAS property wealth, regional land shares, corporate
allocation), which would require re-running the imputation per draw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from extensions import load_baseline, _gini, _deciles  # noqa: E402

OUTPUT = ROOT / "results" / "bootstrap_ci.json"
SEED = 20260729


def _wmean(x, w):
    return float(np.sum(x * w) / np.sum(w))


def _concentration(x, rank, w):
    order = np.argsort(rank)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    total = cxw[-1]
    return float(1 - np.sum((cxw + np.concatenate([[0], cxw[:-1]])) * w) / (total * cw[-1]))


def statistics(df, w):
    """All headline statistics under weight vector ``w``."""
    ct = df["council_tax_net"].values
    land = df["land"].values
    rate = float(np.sum(ct * w) / np.sum(land * w))
    delta = ct - rate * land

    pw = w * df["people"].values
    inc_d = _deciles(df["net_income"].values, w)
    wlt_d = _deciles(df["total_wealth"].values, w)

    # Poverty: baseline-fixed thresholds, HBAI person shares.
    eq_bhc = df["equivalisation_bhc"].values
    eq_ahc = df["equivalisation_ahc"].values
    y_bhc0 = df["equiv_hbai_bhc"].values
    y_ahc0 = df["equiv_hbai_ahc"].values
    line_bhc = df["poverty_line_bhc"].values / np.where(eq_bhc > 0, eq_bhc, 1)
    line_ahc = df["poverty_line_ahc"].values / np.where(eq_ahc > 0, eq_ahc, 1)
    y_bhc1 = y_bhc0 + delta / np.where(eq_bhc > 0, eq_bhc, 1)
    y_ahc1 = y_ahc0 + delta / np.where(eq_ahc > 0, eq_ahc, 1)

    def pov(y, line):
        return 100 * float(np.sum((y < line) * pw) / np.sum(pw))

    gini0 = _gini(y_bhc0, pw)
    gini1 = _gini(y_bhc1, pw)

    # Wealth-ranked Kakwani for both taxes.
    wealth = df["total_wealth"].values
    g_wealth = _gini(np.maximum(wealth, 0), w)
    rank_w = wealth + np.random.default_rng(0).uniform(0, 1e-6, len(wealth))
    kak_ct = _concentration(ct, rank_w, w) - g_wealth
    kak_lvt = _concentration(rate * land, rank_w, w) - g_wealth

    return {
        "neutral_rate_pct": 100 * rate,
        "pct_winners": 100 * float(np.sum((delta > 1) * w) / np.sum(w)),
        "income_decile_1_mean": _wmean(delta[inc_d == 1], w[inc_d == 1]),
        "income_decile_10_mean": _wmean(delta[inc_d == 10], w[inc_d == 10]),
        "wealth_decile_10_mean": _wmean(delta[wlt_d == 10], w[wlt_d == 10]),
        "poverty_bhc_change_pp": pov(y_bhc1, line_bhc) - pov(y_bhc0, line_bhc),
        "poverty_ahc_change_pp": pov(y_ahc1, line_ahc) - pov(y_ahc0, line_ahc),
        "gini_change": gini1 - gini0,
        "kakwani_wealth_ct": kak_ct,
        "kakwani_wealth_lvt": kak_lvt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()

    _, df = load_baseline()
    w0 = df["weight"].values
    point = statistics(df, w0)

    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in point}
    for b in range(args.reps):
        wb = w0 * rng.poisson(1.0, size=len(w0))
        s = statistics(df, wb)
        for k, v in s.items():
            draws[k].append(v)
        if (b + 1) % 100 == 0:
            print(f"replicate {b + 1}/{args.reps}")

    out = {
        "reps": args.reps,
        "method": "Poisson(1) weight bootstrap; budget-neutral rate re-solved "
        "per replicate; percentile intervals. Captures survey sampling "
        "variation only, not land-value imputation variance.",
        "statistics": {},
    }
    for k, v in draws.items():
        arr = np.array(v)
        out["statistics"][k] = {
            "point": round(point[k], 4),
            "se": round(float(np.std(arr, ddof=1)), 4),
            "ci95": [
                round(float(np.percentile(arr, 2.5)), 4),
                round(float(np.percentile(arr, 97.5)), 4),
            ],
        }
    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTPUT}")
    print(json.dumps(out["statistics"], indent=1))


if __name__ == "__main__":
    main()
