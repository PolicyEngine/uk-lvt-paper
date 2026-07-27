"""Constituency map of the council-tax-to-budget-neutral-LVT swap.

Runs baseline and reform PolicyEngine UK simulations on the Enhanced FRS
2023-24 microdata, computes each household's net income change, and averages
it into the 650 Westminster (2024) parliamentary constituencies using the
per-constituency household weight matrix from policyengine-uk-data
(parliamentary_constituency_weights.h5): avg_i = (W_i @ change) / (W_i @ 1).

Outputs (results/geo/):
  constituency_income_change.csv  code, name, avg_change, households
  map_income_change.png           choropleth, PolicyEngine diverging palette

Follows the map conventions of uk-ai-study/analysis/geo_choropleth.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import figstyle  # noqa: E402

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

YEAR = 2026
WEIGHTS_YEAR = 2025  # only year present in the weights file
# The budget-neutral rate is solved on whichever dataset this script runs on.
# Hard-coding the headline rate from the paper's dataset produced a map on
# which almost every constituency lost, because the swap was not revenue
# neutral on this one.
LVT_RATE: float | None = None

UK_DATA_STORAGE = Path(
    "/Users/janansadeqian/policyengine-uk-data/policyengine_uk_data/storage"
)
WEIGHTS_PATH = UK_DATA_STORAGE / "parliamentary_constituency_weights.h5"
CONSTITUENCY_CSV = UK_DATA_STORAGE / "constituencies_2024.csv"
BOUNDARIES = Path(
    "/Users/janansadeqian/energy-price-shock/dashboard/public/data/"
    "uk_constituencies_2024.geojson"
)

# The constituency weight matrix (650 x 52576) was calibrated against the
# enhanced_frs_2024_25 dataset in the same policyengine-uk-data checkout;
# the pinned HF enhanced_frs_2023_24@1.56.14 has 53508 households and does
# NOT match, so we run on the matching local dataset to keep household
# ordering consistent with the weights.
DATASET_PATH = UK_DATA_STORAGE / "enhanced_frs_2024_25.h5"

GEO = ROOT / "results" / "geo"


def get_dataset():
    from policyengine_uk.data import UKSingleYearDataset

    return UKSingleYearDataset(str(DATASET_PATH))


def budget_neutral_rate(baseline) -> float:
    """Net council tax revenue divided by total land value, on this dataset."""
    w = np.asarray(baseline.calculate("household_weight", YEAR).values, dtype=np.float64)
    ct = np.asarray(
        baseline.calculate("council_tax", YEAR).values, dtype=np.float64
    ) - np.asarray(
        baseline.calculate(
            "council_tax_benefit", YEAR, map_to="household"
        ).values,
        dtype=np.float64,
    )
    land = np.asarray(baseline.calculate("land_value", YEAR).values, dtype=np.float64)
    return float(np.sum(ct * w) / np.sum(land * w))


def household_income_change(dataset: str) -> tuple[np.ndarray, float]:
    from policyengine_uk import Microsimulation

    baseline = Microsimulation(dataset=dataset)
    global LVT_RATE
    if LVT_RATE is None:
        LVT_RATE = budget_neutral_rate(baseline)
        print(f"budget-neutral rate on this dataset: {LVT_RATE:.4%}")
    reform = Microsimulation(
        dataset=dataset,
        reform={
            "gov.contrib.abolish_council_tax": {"2026-01-01.2030-12-31": True},
            "gov.contrib.ubi_center.land_value_tax.rate": {
                "2026-01-01.2030-12-31": LVT_RATE
            },
        },
    )
    base = np.asarray(
        baseline.calculate("household_net_income", YEAR).values, dtype=np.float64
    )
    ref = np.asarray(
        reform.calculate("household_net_income", YEAR).values, dtype=np.float64
    )
    change = ref - base
    w = np.asarray(baseline.calculate("household_weight", YEAR).values, dtype=np.float64)
    aggregate = float(np.sum(change * w))
    mean = aggregate / float(np.sum(w))
    print(f"aggregate net change: £{aggregate / 1e9:.2f}bn (£{mean:.0f} per household)")
    if abs(mean) > 25:
        raise RuntimeError(
            f"Swap is not revenue neutral on this dataset: mean change £{mean:.0f} "
            "per household. The map would show a level shift, not redistribution."
        )
    return change, LVT_RATE


def constituency_table(change: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(CONSTITUENCY_CSV)
    with h5py.File(WEIGHTS_PATH, "r") as f:
        weights = np.asarray(f[str(WEIGHTS_YEAR)], dtype=np.float64)
    if weights.shape != (len(df), change.shape[0]):
        raise RuntimeError(
            f"Shape mismatch: weights {weights.shape}, "
            f"{len(df)} constituencies, {change.shape[0]} households."
        )
    households = weights.sum(axis=1)
    avg = (weights @ change) / np.where(households > 0, households, np.nan)
    return pd.DataFrame(
        {
            "code": df["code"],
            "name": df["name"],
            "avg_change": avg.round(2),
            "households": households.round(0).astype(int),
        }
    )


def draw_map(df: pd.DataFrame, path: Path, rate: float) -> None:
    figstyle.apply_style()
    gdf = gpd.read_file(BOUNDARIES)[["GSScode", "geometry"]]
    merged = gdf.merge(df, left_on="GSScode", right_on="code", how="inner")
    missing = len(df) - len(merged)
    if missing:
        print(f"warning: {missing} constituencies did not join to a boundary")
    # GeoJSON is mislabelled EPSG:4326 but holds British National Grid coords.
    merged = merged.set_crs(27700, allow_override=True)

    vals = merged["avg_change"].to_numpy()
    vmax = float(np.nanpercentile(np.abs(vals), 95)) or float(np.abs(vals).max())
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    merged.plot(
        column="avg_change",
        cmap=figstyle.DIVERGING,
        norm=norm,
        ax=ax,
        edgecolor="white",
        linewidth=0.15,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=figstyle.DIVERGING)
    cbar = fig.colorbar(
        sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, extend="both"
    )
    cbar.outline.set_visible(False)
    cbar.set_label(
        "Average household net income change (£/year), "
        f"council tax → {rate:.2%} LVT"
    )
    figstyle.save(fig, path)
    print("wrote", path)


def main() -> None:
    GEO.mkdir(parents=True, exist_ok=True)
    dataset = get_dataset()
    change, rate = household_income_change(dataset)
    table = constituency_table(change)
    csv_path = GEO / "constituency_income_change.csv"
    table.to_csv(csv_path, index=False)
    print("wrote", csv_path)

    print("\nBiggest gains:")
    print(table.nlargest(5, "avg_change")[["name", "avg_change"]].to_string(index=False))
    print("\nBiggest losses:")
    print(table.nsmallest(5, "avg_change")[["name", "avg_change"]].to_string(index=False))

    draw_map(table, GEO / "map_income_change.png", rate)


if __name__ == "__main__":
    main()
