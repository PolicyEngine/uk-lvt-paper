"""Authority codes: the English universe, the MHCLG crosswalk and Isles of Scilly.

The universe is the 296 English local authorities (ONS codes beginning
E06, E07, E08 or E09) listed in VOA Council Tax stock of properties 2026,
table CTSOP1.0, at 31 March 2026. The latest new billing authorities in the
CTSOP notes date from 1 April 2023 (Cumberland, Westmorland and Furness,
North Yorkshire), and the 295 English authorities in UK HPI carry the same
codes, so these are referred to as 2023 codes here.

MHCLG's land value estimates 2023 publish Barnsley and Sheffield under their
pre-April-2023 codes; UK HPI and CTSOP use the 2023 codes. Neither MHCLG nor
UK HPI publishes the Isles of Scilly.
"""

from __future__ import annotations

import pandas as pd

N_ENGLISH_AUTHORITIES = 296
ENGLISH_LA_PREFIXES = ("E06", "E07", "E08", "E09")
ENGLAND_CODE = "E92000001"

# MHCLG code -> April 2023 code.
MHCLG_TO_2023 = {
    "E08000016": "E08000038",  # Barnsley
    "E08000019": "E08000039",  # Sheffield
}

ISLES_OF_SCILLY = "E06000053"
CORNWALL = "E06000052"
# Authority with no MHCLG or UK HPI value -> donor whose values it takes.
IMPUTATION_DONORS = {ISLES_OF_SCILLY: CORNWALL}


def is_english_la(code: str) -> bool:
    return isinstance(code, str) and code.startswith(ENGLISH_LA_PREFIXES)


def to_2023_code(code: str) -> str:
    """Map an MHCLG authority code to its April 2023 code (identity otherwise)."""
    return MHCLG_TO_2023.get(code, code)


def crosswalk_mhclg(frame: pd.DataFrame, code_col: str = "la_code_mhclg") -> pd.DataFrame:
    """Add ``la_code`` (2023 code) next to the MHCLG code column."""
    out = frame.copy()
    out["la_code"] = out[code_col].map(to_2023_code)
    return out


def check_universe(codes: pd.Series, expected_n: int | None = N_ENGLISH_AUTHORITIES) -> None:
    """Raise unless ``codes`` are unique English authority codes on 2023 codes.

    ``expected_n`` is 296 for the real universe; tests pass ``None`` to check
    synthetic universes of any size.
    """
    codes = pd.Series(codes)
    problems = []
    if expected_n is not None and len(codes) != expected_n:
        problems.append(f"{len(codes)} rows, expected {expected_n}")
    dup = codes[codes.duplicated()].unique().tolist()
    if dup:
        problems.append(f"duplicated codes {dup}")
    bad = [c for c in codes if not is_english_la(c)]
    if bad:
        problems.append(f"non-English or malformed codes {bad[:5]}")
    stale = sorted(set(codes) & set(MHCLG_TO_2023))
    if stale:
        problems.append(f"pre-2023 codes present {stale}")
    if problems:
        raise ValueError("authority universe invalid: " + "; ".join(problems))
