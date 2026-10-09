"""Local-authority land values for England (prototype).

Part of the constituency-level LVT analysis. This package is self-contained:
nothing in the paper pipeline (``uk_lvt.pipeline_direct`` and friends) imports
it, and it changes no model parameter or published number.

Modules
-------
``manifest``
    Loads and validates ``sources.yaml``, the input manifest (URL, publisher,
    edition, licence, retrieval date, sha256 for every input).
``fetch``
    Downloads pinned inputs into a git-ignored cache and verifies their
    sha256. ``python -m uk_lvt.land.fetch --help``.
``derive``
    Turns cached raw files into the small committed tables in ``data/``.
``geography``
    The 296-authority universe, the MHCLG code crosswalk and the Isles of
    Scilly rule.
``lambda_raw``
    Raw land share per authority (MHCLG plot land per new dwelling divided
    by a stock price) and its sensitivities. ``python -m
    uk_lvt.land.lambda_raw`` rebuilds ``results/land/``.

See ``docs/land_methods.md`` for the method, every assumption, and the steps
not yet done.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
MANIFEST_PATH = PACKAGE_DIR / "sources.yaml"
DERIVED_DIR = PACKAGE_DIR / "data"
DERIVED_MANIFEST_PATH = DERIVED_DIR / "derived_manifest.json"
RESULTS_DIR = REPO_ROOT / "results" / "land"
METHODS_NOTE_PATH = REPO_ROOT / "docs" / "land_methods.md"
