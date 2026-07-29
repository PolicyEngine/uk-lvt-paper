"""Monte Carlo perturbation of within-region land shares.

The central specification applies a single land share lambda_r to every
property in region r (equation eq:land). MHCLG's local-authority estimates
show substantial within-region dispersion that this suppresses. Absent
household-level land shares, this script bounds the consequence
parametrically: each draw multiplies every household's land share by an
independent mean-preserving lognormal factor with coefficient of variation
CV, truncates the resulting share to [0.05, 0.95], re-solves the
budget-neutral rate and recomputes the headline statistics. CVs of 0.1, 0.2
and 0.3 bracket the within-region dispersion visible in the MHCLG
local-authority land value estimates.

Only the household residential component (lambda_r * P_i) is perturbed;
directly owned land A_i and allocated corporate land are unchanged.

Writes ``results/lambda_mc.json``.

Run:  python analysis/lambda_mc.py               (100 draws per CV)
      python analysis/lambda_mc.py --draws 20    (quick check)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from extensions import load_baseline, _deciles  # noqa: E402

OUTPUT = ROOT / "results" / "lambda_mc.json"
SEED = 20260729
CVS = (0.1, 0.2, 0.3)

# Regional land shares, matching methodology table tab:lambda.
LAMBDA = {
    "LONDON": 0.85, "SOUTH_EAST": 0.65, "EAST_OF_ENGLAND": 0.60,
    "SOUTH_WEST": 0.58, "WEST_MIDLANDS": 0.52, "EAST_MIDLANDS": 0.48,
    "NORTH_WEST": 0.47, "WALES": 0.47, "YORKSHIRE": 0.46,
    "NORTH_EAST": 0.42, "SCOTLAND": 0.44, "NORTHERN_IRELAND": 0.44,
}


def headline(df, delta, w):
    inc_d = df["income_decile"].values  # model equivalised-income deciles
    wlt_d = _deciles(df["total_wealth"].values, w)

    def dmean(d, k):
        m = d == k
        return float(np.sum(delta[m] * w[m]) / np.sum(w[m]))

    return {
        "pct_winners": 100 * float(np.sum((delta > 1) * w) / np.sum(w)),
        "income_decile_1_mean": dmean(inc_d, 1),
        "wealth_decile_10_mean": dmean(wlt_d, 10),
    }


def region_lambda(df):
    regions = df["region"].values
    lam = np.zeros(len(df))
    for key, val in LAMBDA.items():
        lam[np.char.find(regions.astype(str), key) >= 0] = val
    missing = lam == 0
    if missing.any():
        unmatched = sorted(set(df["region"].values[missing]))
        raise SystemExit(f"unmatched regions: {unmatched} — update LAMBDA keys")
    return lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=100)
    args = ap.parse_args()

    _, df = load_baseline()
    w = df["weight"].values
    ct = df["council_tax_net"].values
    corp = df["corporate_land"].values
    hh_land = df["household_land"].values
    owned = df["owned_land"].values
    lam = region_lambda(df)
    # Residential component of household land: hh_land = lam*P + owned_land.
    resid = np.maximum(hh_land - owned, 0.0)
    prop_val = np.where(lam > 0, resid / lam, 0.0)  # implied lambda_r * P_i base

    point_land = resid + owned + corp
    rate0 = float(np.sum(ct * w) / np.sum(point_land * w))
    out = {
        "central": {"rate_pct": round(100 * rate0, 3),
                    **{k: round(v, 1) for k, v in
                       headline(df, ct - rate0 * point_land, w).items()}},
        "note": "mean-preserving lognormal perturbation of household land "
        "shares, truncated to [0.05, 0.95]; residential component only",
        "cv_results": {},
    }

    rng = np.random.default_rng(SEED)
    for cv in CVS:
        sigma = np.sqrt(np.log(1 + cv ** 2))
        stats = {"rate_pct": [], "pct_winners": [],
                 "income_decile_1_mean": [], "wealth_decile_10_mean": []}
        for _ in range(args.draws):
            factor = rng.lognormal(-0.5 * sigma ** 2, sigma, size=len(df))
            lam_i = np.clip(lam * factor, 0.05, 0.95)
            land = lam_i * prop_val + owned + corp
            rate = float(np.sum(ct * w) / np.sum(land * w))
            delta = ct - rate * land
            h = headline(df, delta, w)
            stats["rate_pct"].append(100 * rate)
            for k, v in h.items():
                stats[k].append(v)
        out["cv_results"][str(cv)] = {
            k: {"mean": round(float(np.mean(v)), 2),
                "sd": round(float(np.std(v, ddof=1)), 2),
                "p2.5": round(float(np.percentile(v, 2.5)), 2),
                "p97.5": round(float(np.percentile(v, 97.5)), 2)}
            for k, v in stats.items()
        }
        print(cv, out["cv_results"][str(cv)])

    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
