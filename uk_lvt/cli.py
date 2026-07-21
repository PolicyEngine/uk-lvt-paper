"""Command-line entry point: ``uk-lvt-build``.

Runs the full PolicyEngine pipeline (requires licensed Enhanced FRS data)
and writes ``results/lvt_results.json``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_YEAR,
    generate_results_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the land value tax study results JSON."
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--uk-data-root",
        type=Path,
        help="Optional path to a local policyengine-uk-data checkout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = generate_results_file(
        year=args.year,
        output_path=args.output,
        uk_data_root=args.uk_data_root,
    )
    print(f"Results saved to {args.output}")
    print(
        "Summary: "
        f"total land £{results['baseline']['total_land_tn']}tn, "
        f"budget-neutral LVT rate {results['council_tax_replacement']['required_lvt_rate_pct']}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
