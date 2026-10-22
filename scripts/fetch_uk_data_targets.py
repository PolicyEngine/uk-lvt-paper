"""Fetch the ONS land target files from policyengine-uk-data at a pinned commit.

``uk_lvt.pipeline_direct`` reads the ONS National Balance Sheet land targets
from policyengine-uk-data (``ons_comparison`` in results/lvt_results.json).
That package is not on PyPI at the release the paper used, and installing it
pulls in its whole data-build stack, so this script downloads only the three
source files the loader needs, at the commit tagged ``1.56.14``, checks their
SHA-256 against the values below, and prints the directory to pass as
``--uk-data-root``.

Usage::

    uv run python scripts/fetch_uk_data_targets.py [destination]
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "PolicyEngine/policyengine-uk-data"
COMMIT = "ebf733c6ee7c0cea0afa7098bf4fbbe9922814bb"  # tag 1.56.14
FILES = {
    "policyengine_uk_data/targets/schema.py": (
        "9560b3910694caed1f2a9bb219b4da301afaee2dfd088cd1cee9a5b76bfd21e1"
    ),
    "policyengine_uk_data/targets/sources/_land.py": (
        "ce80dc27b570e70ae9d3ba98bb75f714fbc3aebcd78f1f7329b0d8ef955b7907"
    ),
    "policyengine_uk_data/targets/sources/ons_land_values.py": (
        "cd557c50feabbc6b5e07ee9dd9dd74770e3d2c1fdadb0692d1a47ccb6db79bde"
    ),
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = ROOT / ".cache" / f"policyengine-uk-data-{COMMIT[:12]}"


def fetch(dest: Path = DEFAULT_DEST) -> Path:
    for relative, expected in FILES.items():
        target = dest / relative
        if (
            target.is_file()
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        ):
            continue
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {url}: {actual} != {expected}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return dest


if __name__ == "__main__":
    print(fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DEST))
