"""``analysis/geo_map.py`` refuses to run before any optional import.

The script is superseded (see ``results/geo/README.md``). In a clean locked
environment its imports (h5py from the ``simulation`` extra, and geopandas,
which no extra declares) are missing, so the refusal has to come first or the
user sees an ImportError instead of the reason.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEO_MAP = ROOT / "analysis" / "geo_map.py"

# Runs geo_map.py as __main__ with every third-party module it imports made
# unimportable, whatever the current environment has installed.
RUNNER = """
import runpy
import sys

BLOCKED = {"figstyle", "geopandas", "h5py", "matplotlib", "numpy", "pandas"}


class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {name!r} (blocked)")
        return None


sys.meta_path.insert(0, Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _run(*args):
    return subprocess.run(
        [sys.executable, "-c", RUNNER, str(GEO_MAP), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_refuses_with_the_reason_before_any_optional_import():
    result = _run()
    assert result.returncode == 1
    assert "analysis/geo_map.py is superseded" in result.stderr
    assert "Traceback" not in result.stderr


def test_allow_superseded_skips_the_refusal():
    result = _run("--allow-superseded")
    assert result.returncode == 1
    assert "(blocked)" in result.stderr
    assert "is superseded" not in result.stderr
