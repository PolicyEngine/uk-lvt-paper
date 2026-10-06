"""Full-pipeline driver.

Regenerates every committed results JSON from the licensed Enhanced FRS
dataset (requires ``uv sync --frozen --extra simulation`` and access to the
dataset; see README "Reproduce"):

1. ``uk_lvt.pipeline_direct`` -> ``results/lvt_results.json`` (canonical);
2. ``analysis/extensions.py`` -> ``results/extensions.json``;
3. ``analysis/deep_dive.py`` -> ``results/deep_dive.json``;
4. ``analysis/submission_extras.py`` -> ``results/submission_extras.json``;
5. with ``--with-resampling``, ``analysis/bootstrap_ci.py`` and
   ``analysis/lambda_mc.py`` -> ``results/bootstrap_ci.json`` and
   ``results/lambda_mc.json``.

Every script reads the dataset named by ``--dataset`` (exported as
``UK_LVT_DATASET``), defaulting to the pinned Hugging Face file. The committed
JSONs let ``figures.py`` reproduce every chart without this step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECONDARY = ("extensions.py", "deep_dive.py", "submission_extras.py")
RESAMPLING = ("bootstrap_ci.py", "lambda_mc.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset path or hf:// URL (default: $UK_LVT_DATASET or the pin).",
    )
    parser.add_argument(
        "--uk-data-root",
        type=Path,
        default=None,
        help="Local policyengine-uk-data checkout holding the ONS land targets.",
    )
    parser.add_argument(
        "--with-resampling",
        action="store_true",
        help="Also rerun the bootstrap and land-share Monte Carlo (slow).",
    )
    args = parser.parse_args(argv)

    env = dict(os.environ)
    if args.dataset:
        env["UK_LVT_DATASET"] = args.dataset

    pipeline_cmd = [sys.executable, "-m", "uk_lvt.pipeline_direct"]
    if args.uk_data_root is not None:
        pipeline_cmd += ["--uk-data-root", str(args.uk_data_root)]
    commands = [pipeline_cmd]
    scripts = SECONDARY + (RESAMPLING if args.with_resampling else ())
    commands += [[sys.executable, str(ROOT / "analysis" / s)] for s in scripts]

    for cmd in commands:
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
