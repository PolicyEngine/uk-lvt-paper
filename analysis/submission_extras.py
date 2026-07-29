"""Additional analyses for journal submission.

Writes ``results/submission_extras.json`` with three blocks, all computed on
the same validated arithmetic as ``extensions.py`` (delta = council tax saved
minus LVT; no benefit interactions):

``pass_through_curve``
    The 50 per cent rent pass-through variant in ``extensions.py``
    generalised to theta in {0, 0.25, 0.5, 0.75, 1}: the share of the LVT on
    privately rented dwellings shifted to tenants, with the same aggregate
    rebated to owner households in proportion to household land.

``joint_income_wealth``
    10x10 matrix of the mean net income change by (model equivalised income
    decile) x (equal-weight wealth decile), with cell population shares --
    the direct exhibit of the income/wealth incidence dissociation.

``reynolds_smolensky``
    Reynolds-Smolensky indices (Gini before the tax minus Gini after) on
    person-weighted equivalised household net income, for net council tax,
    the budget-neutral LVT, and the swap itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from extensions import (  # noqa: E402
    load_baseline,
    summarise_delta,
    _gini,
    _wmean,
)

OUTPUT = ROOT / "results" / "submission_extras.json"
THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def pass_through_curve(df, central_rate):
    """Incidence under tenant pass-through share theta."""
    w = df["weight"].values
    hh_land = df["household_land"].values
    land = df["land"].values
    ct_saved = df["council_tax_net"].values

    private_renter = df["tenure"].values == "RENT_PRIVATELY"
    owner = np.isin(df["tenure"].values, ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"])
    proxy = np.zeros(len(df))
    for region in np.unique(df["region"].values):
        rmask = df["region"].values == region
        owners_r = rmask & owner
        if owners_r.any():
            proxy[rmask & private_renter] = _wmean(hh_land[owners_r], w[owners_r])
    owner_land_total = float(np.sum(hh_land[owner] * w[owner]))

    rows = []
    for theta in THETAS:
        tenant_charge = theta * central_rate * proxy
        total_shifted = float(np.sum(tenant_charge * w))
        rebate = np.where(owner, hh_land / owner_land_total * total_shifted, 0.0)
        delta = ct_saved - central_rate * land - tenant_charge + rebate
        row = summarise_delta(
            df,
            delta,
            f"theta={theta:.2f}",
            rate_pct=round(central_rate * 100, 3),
            theta=theta,
            amount_shifted_to_tenants_bn=round(total_shifted / 1e9, 1),
        )
        m = private_renter
        row["private_renter_mean_change"] = round(_wmean(delta[m], w[m]))
        rows.append(row)
    return rows


def joint_income_wealth(df, delta):
    """Mean net change and population share by income x wealth decile."""
    w = df["weight"].values
    inc = df["income_decile"].values.astype(int)
    wea = df["wealth_decile_own"].values.astype(int)
    total_w = float(np.sum(w))
    mean_change = [[None] * 10 for _ in range(10)]
    pop_share = [[0.0] * 10 for _ in range(10)]
    for i in range(1, 11):
        for j in range(1, 11):
            m = (inc == i) & (wea == j)
            pop_share[i - 1][j - 1] = round(100 * float(np.sum(w[m])) / total_w, 2)
            if m.any():
                mean_change[i - 1][j - 1] = round(_wmean(delta[m], w[m]))
    # Weighted rank correlation between income and wealth (Spearman on
    # weighted fractional ranks), the summary statistic of the dissociation.
    def frac_rank(x):
        order = np.argsort(x)
        r = np.empty(len(x))
        cw = np.cumsum(w[order]) - 0.5 * w[order]
        r[order] = cw / total_w
        return r

    ri = frac_rank(df["net_income"].values)
    rw = frac_rank(df["total_wealth"].values)
    cov = np.sum(w * (ri - _wmean(ri, w)) * (rw - _wmean(rw, w))) / total_w
    sd = np.sqrt(
        np.sum(w * (ri - _wmean(ri, w)) ** 2)
        * np.sum(w * (rw - _wmean(rw, w)) ** 2)
    ) / total_w
    return {
        "rows": "income decile 1-10 (model equivalised income deciles)",
        "cols": "wealth decile 1-10 (equal-weight total wealth deciles)",
        "mean_net_change": mean_change,
        "population_share_pct": pop_share,
        "rank_correlation_income_wealth": round(float(cov / sd), 3),
    }


def reynolds_smolensky(df, central_rate):
    """RS = Gini(income before the tax) - Gini(income after), person-weighted
    equivalised income, HBAI BHC concept."""
    pw = df["person_weight"].values
    eq = df["equivalisation_bhc"].values
    y_post_ct = df["equiv_hbai_bhc"].values  # baseline: net of council tax
    ct = df["council_tax_net"].values / np.where(eq > 0, eq, 1)
    lvt = central_rate * df["land"].values / np.where(eq > 0, eq, 1)
    y_pre = y_post_ct + ct  # income before either tax
    g_pre = _gini(y_pre, pw)
    return {
        "income_concept": "equivalised household net income, person-weighted, "
        "BHC; 'pre' adds back net council tax",
        "gini_pre_tax": round(g_pre, 4),
        "rs_council_tax": round(g_pre - _gini(y_pre - ct, pw), 4),
        "rs_lvt": round(g_pre - _gini(y_pre - lvt, pw), 4),
        "rs_swap_vs_baseline": round(
            _gini(y_post_ct, pw) - _gini(y_pre - lvt, pw), 4
        ),
    }


def main():
    sim, df = load_baseline()
    w = df["weight"].values
    ct_saved = df["council_tax_net"].values
    land = df["land"].values
    target = float(np.sum(ct_saved * w))
    central_rate = target / float(np.sum(land * w))
    delta = ct_saved - central_rate * land

    out = {
        "central_rate_pct": round(central_rate * 100, 3),
        "replacement_target_bn": round(target / 1e9, 1),
        "pass_through_curve": pass_through_curve(df, central_rate),
        "joint_income_wealth": joint_income_wealth(df, delta),
        "reynolds_smolensky": reynolds_smolensky(df, central_rate),
    }
    OUTPUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUTPUT}")
    print(json.dumps({k: out[k] for k in ("central_rate_pct", "reynolds_smolensky")}, indent=1))
    for r in out["pass_through_curve"]:
        print(r["theta"], r["pct_winners"], r["poverty_ahc_change"], r["poverty_bhc_change"])


if __name__ == "__main__":
    main()
