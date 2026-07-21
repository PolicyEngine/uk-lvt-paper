#!/usr/bin/env bash
# Regenerate results and all paper figures.
#
# Stage 1 (optional, licensed data required): rerun the PolicyEngine
# pipeline with RUN_PIPELINE=1. Otherwise the committed
# results/lvt_results.json is used as-is.
# Stage 2: rebuild all figures + CSVs in results/figures/.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${RUN_PIPELINE:-0}" == "1" ]]; then
    python analysis/run_all.py
fi

python analysis/figures.py
echo "Done."
