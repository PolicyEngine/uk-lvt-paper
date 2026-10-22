"""Stock weights and district geography for the transaction residual check.

The stock source is VOA's 2026 Council Tax stock of properties, CTSOP3.0.
It distinguishes bungalows from houses; Price Paid instead distinguishes
detached, semi-detached, terraced and flats. The conversion below preserves
each authority's published CTSOP1.0 dwelling total while explicitly imputing
the categories that cannot be matched. Counts rounded by VOA are not exact.

Price Paid district names, rather than postal towns or counties, identify
the transaction's authority. Historical districts are combined into the
same April 2023 geography as the prototype. This covers the reorganisations
affecting the 2018 onwards transaction window; older boundaries are not
supported. Unrecognised names return None and must be counted by the caller.
No fuzzy name matching or postcode-to-authority inference is performed.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd

from . import DERIVED_DIR

BANDS = tuple("ABCDEFGH")
PROPERTY_TYPES = ("D", "S", "T", "F")
UNIVERSE_PATH = DERIVED_DIR / "voa_ctsop1_2026_england_la.csv"

# ONS's account of the structural changes:
# https://www.ons.gov.uk/methodology/geography/ukgeographies/
# administrativegeography/ourchanginggeography/localgovernmentrestructuring
# The GOV.UK record also covers the 2019 Suffolk and Somerset district mergers:
# https://www.gov.uk/guidance/local-government-structure-and-elections
# Explicit predecessor lists follow the structural changes orders named below.
HISTORICAL_DISTRICT_GROUPS = {
    # Cumbria (Structural Changes) Order 2022, SI 2022/331, articles 2-4.
    "E06000063": ("ALLERDALE", "CARLISLE", "COPELAND"),
    "E06000064": ("BARROW-IN-FURNESS", "EDEN", "SOUTH LAKELAND"),
    # North Yorkshire (Structural Changes) Order 2022, SI 2022/328, article 4.
    # York remains a separate authority.
    "E06000065": (
        "CRAVEN", "HAMBLETON", "HARROGATE", "RICHMONDSHIRE", "RYEDALE",
        "SCARBOROUGH", "SELBY",
    ),
    # Somerset (Structural Changes) Order 2022, SI 2022/329. Taunton Deane
    # and West Somerset merged as Somerset West and Taunton in April 2019.
    # Bath and North East Somerset and North Somerset remain separate.
    "E06000066": (
        "MENDIP", "SEDGEMOOR", "SOUTH SOMERSET", "SOMERSET WEST AND TAUNTON",
        "TAUNTON DEANE", "WEST SOMERSET",
    ),
    # Bournemouth, Dorset and Poole (Structural Changes) Order 2018.
    # Christchurch belongs to BCP, not the new Dorset authority.
    "E06000058": ("BOURNEMOUTH", "CHRISTCHURCH", "POOLE"),
    "E06000059": (
        "EAST DORSET", "NORTH DORSET", "PURBECK", "WEST DORSET",
        "WEYMOUTH AND PORTLAND",
    ),
    # Buckinghamshire (Structural Changes) Order 2019, SI 2019/957.
    # Milton Keynes remains separate.
    "E06000060": ("AYLESBURY VALE", "CHILTERN", "SOUTH BUCKS", "WYCOMBE"),
    # Northamptonshire (Structural Changes) Order 2020, SI 2020/156.
    "E06000061": ("CORBY", "EAST NORTHAMPTONSHIRE", "KETTERING", "WELLINGBOROUGH"),
    "E06000062": ("DAVENTRY", "NORTHAMPTON", "SOUTH NORTHAMPTONSHIRE"),
    # East and West Suffolk (Local Government Changes) Orders 2018,
    # SI 2018/640 and SI 2018/639 respectively, articles 3-5.
    "E07000244": ("SUFFOLK COASTAL", "WAVENEY"),
    "E07000245": ("FOREST HEATH", "ST EDMUNDSBURY"),
}

# Alternative official names observed in the Price Paid district field or
# retained during the 2018 onwards window. These are exact aliases, not
# guesses based on the property's postal town.
DISTRICT_NAME_ALIASES = {
    "CITY OF BRISTOL": "E06000023",
    "BRISTOL": "E06000023",
    "CITY OF DERBY": "E06000015",
    "CITY OF KINGSTON UPON HULL": "E06000010",
    "KINGSTON UPON HULL": "E06000010",
    "CITY OF NOTTINGHAM": "E06000018",
    "CITY OF PETERBOROUGH": "E06000031",
    "CITY OF PLYMOUTH": "E06000026",
    "CITY OF WESTMINSTER": "E09000033",
    "HEREFORDSHIRE": "E06000019",
    "WREKIN": "E06000020",
    "THE WREKIN": "E06000020",
    "SHEPWAY": "E07000112",  # Renamed Folkestone and Hythe in April 2018.
}


def load_authority_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    """Read the prototype's authoritative names, regions and stock totals."""
    return pd.read_csv(path, dtype={"la_code": str})


def normalize_district_name(name: str) -> str:
    """Normalise case, spacing and punctuation without changing name words."""
    name = str(name).upper().replace("&", " AND ")
    name = name.replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", name).split())


def ppd_district_crosswalk(universe: pd.DataFrame | None = None) -> dict[str, str]:
    """Return normalised Price Paid district names mapped to 2023 LA codes.

    Welsh districts intentionally have no entry. The caller should distinguish
    them from unexpected English names when reporting unmatched transactions.
    ``district`` is CSV field 13 (zero-based index 12), after town/city.
    """
    if universe is None:
        universe = load_authority_universe()
    codes = set(universe["la_code"])
    out: dict[str, str] = {}

    def add(name: str, code: str) -> None:
        key = normalize_district_name(name)
        if key in out and out[key] != code:
            raise ValueError(f"ambiguous district alias {name!r}")
        out[key] = code

    for row in universe.itertuples(index=False):
        add(row.la_name, row.la_code)
    for code, names in HISTORICAL_DISTRICT_GROUPS.items():
        if code in codes:
            for name in names:
                add(name, code)
    for name, code in DISTRICT_NAME_ALIASES.items():
        if code in codes:
            add(name, code)
    return out


def ppd_district_to_la(
    district: str, crosswalk: Mapping[str, str] | None = None,
) -> str | None:
    """Map a district to the prototype geography; None means unmatched."""
    if crosswalk is None:
        crosswalk = ppd_district_crosswalk()
    return crosswalk.get(normalize_district_name(district))


_TABLE_NS = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_TABLE = f"{{{_TABLE_NS}}}table"
_ROW = f"{{{_TABLE_NS}}}table-row"
_PARAGRAPH = f"{{{_TEXT_NS}}}p"


def _ods_rows(path: Path, sheet_name: str, n_columns: int) -> Iterator[list[str]]:
    """Read just the relevant ODS cells without an optional spreadsheet engine.

    ODS blank cells may repeat up to the worksheet's column limit; cap expansion
    at the columns used by CTSOP3.0. Clear finished rows to limit memory use.
    """
    selected = found = False
    with ZipFile(path) as archive, archive.open("content.xml") as stream:
        for event, element in ElementTree.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == _TABLE:
                selected = element.get(f"{{{_TABLE_NS}}}name") == sheet_name
                found |= selected
            elif event == "end" and element.tag == _ROW:
                if selected:
                    cells: list[str] = []
                    for cell in element:
                        if len(cells) >= n_columns:
                            break
                        value = " ".join(
                            "".join(p.itertext()) for p in cell.iter(_PARAGRAPH)
                        ).strip()
                        repeats = int(cell.get(f"{{{_TABLE_NS}}}number-columns-repeated", "1"))
                        cells.extend([value] * min(repeats, n_columns - len(cells)))
                    cells.extend([""] * (n_columns - len(cells)))
                    yield cells
                element.clear()
            elif event == "end" and element.tag == _TABLE:
                selected = False
                element.clear()
    if not found:
        raise ValueError(f"ODS sheet {sheet_name!r} not found")


def _stock_count(value: str) -> float:
    value = value.strip().replace(",", "")
    if value == "[c]":
        return 2.5  # Midpoint of nonzero counts below 5; explicitly imputed.
    if value == "[z]":
        return 0.0
    if value.isdigit():
        return float(value)
    raise ValueError(f"unexpected CTSOP3.0 stock cell {value!r}")


def _proportions(values: list[float], fallback: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        values, total = fallback, sum(fallback)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def parse_ctsop3_stock(path: Path, universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert CTSOP3.0 to 296 authorities × four PPD types × eight bands.

    Bungalows are assigned to D/S/T using each authority-band's known house
    composition, falling back to the authority's house totals. Annexes,
    caravans/houseboats/mobile homes and unknown types are distributed across
    D/S/T/F in proportion to the known types after the bungalow allocation.
    Empty bands fall back to the authority composition, then equal shares.

    Suppressed counts [c] are set to 2.5. A final authority-wide scale factor
    reconciles rounding differences with CTSOP1.0 ``all_properties``. The
    resulting fractional stock counts are weights, not asserted exact counts.
    This uses no transaction data in allocating stock categories. Returned
    frame attrs record suppression and reconciliation diagnostics.
    """
    if universe is None:
        universe = load_authority_universe()
    if universe["la_code"].duplicated().any():
        raise ValueError("authority universe has duplicate codes")
    expected = set(universe["la_code"])
    totals = universe.set_index("la_code")["all_properties"].astype(float).to_dict()
    rows = _ods_rows(Path(path), "CTSOP3_0", 49)
    header = next(row for row in rows if row[0].startswith("Geography"))
    names = {
        "D": "Detached House", "S": "Semi-Detached House",
        "T": "Terraced House", "F": "Flat / Maisonette",
        "bungalow": "Bungalow", "annexe": "Annexe",
        "caravan": "Caravan/ Houseboat/ Mobile Home", "unknown": "Unknown [note 17]",
    }
    try:
        columns = {key: header.index(name) for key, name in names.items()}
    except ValueError as error:
        raise ValueError("unexpected CTSOP3.0 header") from error
    source_rows: dict[str, dict[str, dict[str, float]]] = {}
    suppressed_cells = 0
    for row in rows:
        code, band = row[2], row[4]
        if row[0] != "LAUA" or code not in expected or band not in BANDS:
            continue
        authority = source_rows.setdefault(code, {})
        if band in authority:
            raise ValueError(f"duplicate CTSOP3.0 authority-band {code} {band}")
        values = {key: row[index] for key, index in columns.items()}
        suppressed_cells += sum(value == "[c]" for value in values.values())
        authority[band] = {key: _stock_count(value) for key, value in values.items()}
    if set(source_rows) != expected:
        raise ValueError(f"CTSOP3.0 authority mismatch: {sorted(expected - set(source_rows))}")

    records = []
    scale_factors = []
    for code in sorted(expected):
        authority = source_rows[code]
        if set(authority) != set(BANDS):
            raise ValueError(f"CTSOP3.0 {code}: missing bands {set(BANDS) - set(authority)}")
        house_totals = [sum(row[t] for row in authority.values()) for t in PROPERTY_TYPES[:3]]
        house_mix = _proportions(house_totals, [1.0, 1.0, 1.0])
        bungalow_total = sum(row["bungalow"] for row in authority.values())
        type_totals = [
            house_totals[i] + bungalow_total * house_mix[i] for i in range(3)
        ] + [sum(row["F"] for row in authority.values())]
        allocated: dict[str, list[float]] = {}
        for band, row in authority.items():
            band_house_mix = _proportions([row[t] for t in PROPERTY_TYPES[:3]], house_totals)
            counts = [row[t] + row["bungalow"] * band_house_mix[i]
                      for i, t in enumerate(PROPERTY_TYPES[:3])] + [row["F"]]
            unknown = row["annexe"] + row["caravan"] + row["unknown"]
            type_mix = _proportions(counts, type_totals)
            allocated[band] = [count + unknown * weight for count, weight in zip(counts, type_mix)]
        subtotal = sum(sum(counts) for counts in allocated.values())
        if subtotal <= 0 or totals[code] <= 0:
            raise ValueError(f"CTSOP3.0 {code}: nonpositive dwelling total")
        scale = totals[code] / subtotal
        scale_factors.append(scale)
        for band in BANDS:
            for property_type, count in zip(PROPERTY_TYPES, allocated[band]):
                records.append({
                    "la_code": code, "property_type": property_type,
                    "band": band, "stock": count * scale,
                })
    result = pd.DataFrame.from_records(records)
    result.attrs.update({
        "suppressed_cells_imputed": suppressed_cells,
        "suppressed_count_imputation": 2.5,
        "minimum_authority_scale_factor": min(scale_factors),
        "maximum_authority_scale_factor": max(scale_factors),
    })
    return result
