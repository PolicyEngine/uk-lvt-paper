"""Build the small committed tables in ``uk_lvt/land/data/`` from cached raw files.

Usage::

    python -m uk_lvt.land.fetch --only mhclg_lve_2023 ukhpi_average_prices_2026_07 voa_ctsop_2026_summary
    python -m uk_lvt.land.derive

Needs the ``land`` extra (openpyxl for .xlsx, odfpy for .ods). Each output is
recorded in ``data/derived_manifest.json`` with its sha256, row count and the
sha256 of every raw source it was read from, so a committed table can always
be traced to a pinned input.

Outputs
-------
``mhclg_lve2023_residential.csv``
    MHCLG land value estimates 2023, residential sheets, in long form: one row
    per authority x sheet (``floor_adjusted`` = "Residential (adj for floor)",
    ``unadjusted`` = "Residential") x density scenario x price percentile.
    ``land_value_gbp_per_ha`` is empty where MHCLG reports "-" (a negative
    residual value is not published). Codes are as published (pre-2023 codes
    for Barnsley and Sheffield).
``ukhpi_la_average_price.csv``
    UK HPI average price for the English authorities at the dates in
    ``HPI_DATES``, values copied as published.
``voa_ctsop1_2026_england_la.csv``
    CTSOP1.0 property counts by Council Tax band for the 296 English
    authorities, with the region each sits under in the table. ``[c]``
    (suppressed) counts are left empty.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import DERIVED_DIR, DERIVED_MANIFEST_PATH, REPO_ROOT
from .fetch import cached_path, sha256_file
from .geography import N_ENGLISH_AUTHORITIES, is_english_la
from .manifest import load_manifest

MHCLG_SOURCE = "mhclg_lve_2023"
HPI_SOURCE = "ukhpi_average_prices_2026_07"
CTSOP_SOURCE = "voa_ctsop_2026_summary"

MHCLG_SHEETS = {
    "floor_adjusted": "Residential (adj for floor)",
    "unadjusted": "Residential",
}
DENSITY_SCENARIOS = ("low", "medium", "high")
PRICE_PERCENTILES = (25, 40, 50, 60, 75)
HPI_DATES = ("2023-10-01",)

MHCLG_FILE = "mhclg_lve2023_residential.csv"
HPI_FILE = "ukhpi_la_average_price.csv"
CTSOP_FILE = "voa_ctsop1_2026_england_la.csv"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


# --------------------------------------------------------------------------
# MHCLG land value estimates 2023
# --------------------------------------------------------------------------

_EXPECTED_HEADER = [
    "Region", "LA Code", "Local Authority Name", "Low", "Medium", "High",
    "25", "40", "50 (Median)", "60", "75",
    "25", "40", "50 (Median)", "60", "75",
    "25", "40", "50 (Median)", "60", "75",
]


def parse_mhclg_residential(path: Path) -> pd.DataFrame:
    """Parse both residential sheets of the MHCLG 2023 workbook (long form)."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records = []
    for sheet_key, sheet_name in MHCLG_SHEETS.items():
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        header = [str(v).strip() if v is not None else "" for v in rows[4][1:22]]
        if header != _EXPECTED_HEADER:
            raise ValueError(f"{sheet_name}: unexpected header {header}")
        group = [str(v or "") for v in rows[3][7:22:5]]
        if not (group[0].startswith("Low site density")
                and group[1].startswith("Medium site density")
                and group[2].startswith("High site density")):
            raise ValueError(f"{sheet_name}: unexpected density groups {group}")
        for row in rows[5:]:
            code = row[2]
            if code is None:
                break
            region, name = str(row[1]).strip(), str(row[3]).strip()
            densities = row[4:7]
            for d_idx, scenario in enumerate(DENSITY_SCENARIOS):
                dph = densities[d_idx]
                if not isinstance(dph, (int, float)) or dph <= 0:
                    raise ValueError(f"{sheet_name} {code}: bad density {dph!r}")
                for p_idx, pct in enumerate(PRICE_PERCENTILES):
                    v = row[7 + 5 * d_idx + p_idx]
                    if isinstance(v, (int, float)):
                        value = int(v) if float(v).is_integer() else float(v)
                    elif isinstance(v, str) and v.strip() == "-":
                        value = None
                    else:
                        raise ValueError(f"{sheet_name} {code}: unexpected cell {v!r}")
                    records.append({
                        "la_code_mhclg": str(code).strip(),
                        "la_name": name,
                        "region": region,
                        "sheet": sheet_key,
                        "density_scenario": scenario,
                        "dwellings_per_ha": int(dph) if float(dph).is_integer() else float(dph),
                        "price_percentile": pct,
                        "land_value_gbp_per_ha": value,
                    })
    out = pd.DataFrame.from_records(records)
    out["land_value_gbp_per_ha"] = out["land_value_gbp_per_ha"].astype("Int64")
    return out


# --------------------------------------------------------------------------
# UK HPI
# --------------------------------------------------------------------------

def parse_hpi_average_prices(path: Path, dates=HPI_DATES) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"Date", "Region_Name", "Area_Code", "Average_Price"}
    if not expected <= set(raw.columns):
        raise ValueError(f"HPI file missing columns {expected - set(raw.columns)}")
    sel = raw[raw["Area_Code"].map(is_english_la) & raw["Date"].isin(dates)]
    out = sel.rename(columns={
        "Area_Code": "la_code", "Region_Name": "la_name",
        "Date": "date", "Average_Price": "average_price",
    })[["la_code", "la_name", "date", "average_price"]]
    if out.duplicated(["la_code", "date"]).any():
        raise ValueError("HPI file has duplicate authority-date rows")
    return out.sort_values(["date", "la_code"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# VOA CTSOP1.0
# --------------------------------------------------------------------------

_BANDS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _clean_name(name: str) -> str:
    name = re.sub(r"\s*\[note \d+\]", "", str(name)).strip()
    return re.sub(r" UA$", "", name)


def _count(v: str):
    v = str(v).strip()
    if re.fullmatch(r"\d+", v):
        return int(v)
    if v in {"[c]", "[z]", "[x]"}:
        return None
    raise ValueError(f"unexpected CTSOP cell {v!r}")


def parse_ctsop1(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, engine="odf", sheet_name="CTSOP1_0", header=4, dtype=str)
    raw.columns = ["geo", "ba_code", "code", "name", *_BANDS, "I", "all"]
    region_code = region_name = None
    records = []
    for r in raw.itertuples(index=False):
        if r.geo == "REGL":
            region_code, region_name = r.code, r.name.strip()
        elif r.geo == "NATL" and str(r.code).startswith("W"):
            region_code = region_name = None
        elif r.geo == "LAUA" and is_english_la(r.code):
            if region_code is None:
                raise ValueError(f"{r.code} has no region above it")
            rec = {
                "la_code": r.code,
                "la_name": _clean_name(r.name),
                "region_code": region_code,
                "region_name": region_name,
                "billing_authority_code": str(r.ba_code).strip(),
            }
            for b in _BANDS:
                rec[f"band_{b.lower()}"] = _count(getattr(r, b))
            rec["all_properties"] = _count(r.all)
            if rec["all_properties"] is None:
                raise ValueError(f"{r.code}: total properties suppressed")
            records.append(rec)
    out = pd.DataFrame.from_records(records)
    for b in _BANDS:
        out[f"band_{b.lower()}"] = out[f"band_{b.lower()}"].astype("Int64")
    if len(out) != N_ENGLISH_AUTHORITIES or out["la_code"].duplicated().any():
        raise ValueError(f"CTSOP1.0 gave {len(out)} English authorities")
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build_all(cache_dir: Path | None = None, out_dir: Path = DERIVED_DIR) -> dict:
    sources, _ = load_manifest()
    jobs = [
        (MHCLG_FILE, MHCLG_SOURCE, parse_mhclg_residential,
         "MHCLG land value estimates 2023, residential sheets, long form"),
        (HPI_FILE, HPI_SOURCE, parse_hpi_average_prices,
         f"UK HPI average price, English authorities, dates {list(HPI_DATES)}"),
        (CTSOP_FILE, CTSOP_SOURCE, parse_ctsop1,
         "VOA CTSOP1.0 property counts by band, 296 English authorities, 31 March 2026"),
    ]
    manifest = {
        "generated_by": "python -m uk_lvt.land.derive",
        "files": {},
    }
    for fname, sid, parser, desc in jobs:
        raw_path = cached_path(sid, cache_dir)
        frame = parser(raw_path)
        out_path = Path(out_dir) / fname
        _write_csv(frame, out_path)
        manifest["files"][fname] = {
            "description": desc,
            "sha256": sha256_file(out_path),
            "rows": int(len(frame)),
            "columns": list(frame.columns),
            "sources": [{"id": sid, "sha256": sources[sid].sha256}],
        }
    manifest_path = Path(out_dir) / DERIVED_MANIFEST_PATH.name
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    args = ap.parse_args(argv)
    manifest = build_all(args.cache_dir, args.out_dir)
    for fname, meta in manifest["files"].items():
        rel = (Path(args.out_dir) / fname)
        try:
            rel = rel.resolve().relative_to(REPO_ROOT)
        except ValueError:
            pass
        print(f"{rel}  rows={meta['rows']}  sha256={meta['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
