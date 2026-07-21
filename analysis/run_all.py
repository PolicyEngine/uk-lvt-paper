"""Full-pipeline driver.

Runs the PolicyEngine microsimulation (requires a Hugging Face token with
access to the licensed Enhanced FRS dataset) and rewrites
``results/lvt_results.json``. That JSON is committed to the repo, so
``figures.py`` reproduces every chart without this step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from uk_lvt.pipeline import DEFAULT_OUTPUT_PATH, DEFAULT_YEAR, generate_results_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument(
        "--output", type=Path, default=ROOT / DEFAULT_OUTPUT_PATH
    )
    parser.add_argument(
        "--uk-data-root",
        type=Path,
        help="Optional path to a local policyengine-uk-data checkout.",
    )
    args = parser.parse_args()
    results = generate_results_file(
        year=args.year, output_path=args.output, uk_data_root=args.uk_data_root
    )
    print(f"Wrote {args.output}")
    print(
        f"Budget-neutral rate: "
        f"{results['council_tax_replacement']['required_lvt_rate_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
