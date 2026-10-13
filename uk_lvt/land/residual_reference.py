"""Parse open reference inputs for the transaction residual prototype.

Source bytes are verified by the caller's manifest. The TPA configuration is
read as syntax, never executed. EHS floor areas are derived from the published
2018 workbook, including its explicitly imputed detailed house-type split.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from statistics import mean

import pandas as pd


def _positive(value: object, description: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{description} must be finite and positive")
    return result


def parse_tpa_cells(path: Path | str) -> pd.DataFrame:
    """Recover authority land/value ratios from TPA's stock-weighted deciles.

The reference calibrates its price level and rebuilding-cost level nationally;
this function retains those published calibrations without applying them to
our own transaction estimates. Counts weight the cell means, not cell ratios.
"""
    payload = json.loads(Path(path).read_text())
    rows = []
    for code, authority in sorted(payload["las"].items()):
        dwellings = property_value = land_value = 0.0
        for band in authority["bands"].values():
            count = float(band["count"])
            if not math.isfinite(count) or count < 0:
                raise ValueError(f"Invalid TPA dwelling count for {code}")
            values = [float(v) for v in band["value_deciles"]]
            lands = [float(v) for v in band["land_deciles"]]
            if not values or len(values) != len(lands):
                raise ValueError(f"Invalid TPA decile arrays for {code}")
            if any(not math.isfinite(v) or v < 0 for v in values + lands):
                raise ValueError(f"Invalid TPA decile values for {code}")
            dwellings += count
            property_value += count * mean(values)
            land_value += count * mean(lands)
        _positive(property_value, f"TPA property value for {code}")
        if land_value > property_value:
            raise ValueError(f"TPA land value exceeds property value for {code}")
        rows.append({
            "la_code": code,
            "tpa_land_value_share": land_value / property_value,
            "dwellings": dwellings,
            "property_value_gbp": property_value,
            "land_value_gbp": land_value,
        })
    return pd.DataFrame(rows, columns=[
        "la_code", "tpa_land_value_share", "dwellings",
        "property_value_gbp", "land_value_gbp",
    ])


def parse_floor_areas(path: Path | str) -> pd.DataFrame:
    """Derive 36 region/type floor areas from EHS 2018 AT1.2 and AT1.5.

For D/S/T, multiply the national detailed house-type mean (AT1.2) by the
regional house-or-bungalow mean divided by its national mean (AT1.5).
The detailed region/type cross-tab is not published: this is a separability
assumption. F uses AT1.5's directly published regional flat mean. EHS excludes
bedsits; Price Paid cannot separately identify bungalows within D/S/T.
"""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        detailed = workbook["AT1.2"]
        broad = workbook["AT1.5"]
        type_labels = {
            "detached house": "D",
            "semi-detached house": "S",
            "all terraced houses": "T",
        }
        national_types = {}
        for row in detailed.iter_rows(values_only=True):
            if len(row) > 2 and row[1] in type_labels:
                national_types[type_labels[row[1]]] = _positive(
                    row[2], f"National EHS floor area for {row[1]}"
                )
        if set(national_types) != set("DST"):
            raise ValueError("EHS AT1.2 is missing a detailed house-type mean")
        national_broad = None
        regional_rows = []
        for row in broad.iter_rows(values_only=True):
            if len(row) < 6 or not isinstance(row[1], str):
                continue
            label = row[1].strip()
            if label == "all dwellings":
                national_broad = _positive(row[2], "National EHS house/bungalow area")
            elif isinstance(row[2], (float, int)) and isinstance(row[5], (float, int)):
                regional_rows.append((label, row[2], row[5]))
        if national_broad is None or len(regional_rows) != 9:
            raise ValueError("EHS AT1.5 must contain a national mean and nine regions")
        rows = []
        for region, house_area, flat_area in regional_rows:
            region = region.replace("Yorkshire and the Humber", "Yorkshire and The Humber")
            ratio = _positive(house_area, f"EHS house area for {region}") / national_broad
            for property_type in "DST":
                rows.append({
                    "region_name": region,
                    "property_type": property_type,
                    "floor_area_m2": national_types[property_type] * ratio,
                })
            rows.append({
                "region_name": region,
                "property_type": "F",
                "floor_area_m2": _positive(flat_area, f"EHS flat area for {region}"),
            })
        return pd.DataFrame(rows)
    finally:
        workbook.close()


def parse_rebuild_constants(path: Path | str) -> tuple[dict[str, float], dict[str, float]]:
    """Read TPA's open cost assumptions using AST literal evaluation only.

Return costs keyed by D/S/T/F and regional factors keyed by official E12 codes.
National calibration constants in the same file are deliberately not imported.
"""
    wanted = {"BASE_COST_PER_M2", "REGION_FACTOR"}
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                if target.id in found:
                    raise ValueError(f"Duplicate TPA assignment {target.id}")
                found[target.id] = ast.literal_eval(node.value)
    if set(found) != wanted:
        raise ValueError("TPA config is missing rebuilding-cost assumptions")
    costs = {str(k): _positive(v, f"TPA rebuild cost {k}")
             for k, v in found["BASE_COST_PER_M2"].items()}
    factors = {str(k): _positive(v, f"TPA regional factor {k}")
               for k, v in found["REGION_FACTOR"].items()}
    if set(costs) != set("DSTF"):
        raise ValueError("TPA rebuild constants must cover D/S/T/F")
    if set(factors) != {f"E1200000{i}" for i in range(1, 10)}:
        raise ValueError("TPA regional factors must cover nine English regions")
    return costs, factors


def parse_ons_dwelling_land(path: Path | str, year: int = 2021) -> float:
    """Read the historical ONS Figure 1 subclass fallback, in pounds.

The 2022 publication rounds its UK land-under-dwellings series to £0.1tn;
this source is not the latest NBS edition and is not England-only.
"""
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        rows = [[cell.strip() for cell in row] for row in csv.reader(stream)]
    if not any(row[:2] == ["Unit", "£ trillion"] for row in rows):
        raise ValueError("Expected ONS chart values in £ trillion")
    column = None
    for row in rows:
        if "Land underlying dwellings" in row:
            column = row.index("Land underlying dwellings")
        elif column is not None and row and row[0] == str(year):
            return _positive(row[column], "ONS dwelling land") * 1e12
    raise ValueError(f"ONS dwelling-land chart has no observation for {year}")
