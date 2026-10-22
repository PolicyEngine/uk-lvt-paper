#!/usr/bin/env bash
# Regenerate results and all paper figures.
#
# Stage 1 (optional, licensed data required): rerun the canonical pipeline
# (uk_lvt.pipeline_direct) and the secondary generators with RUN_PIPELINE=1.
# Extra arguments are passed to analysis/run_all.py, e.g.
#   RUN_PIPELINE=1 analysis/regenerate_all.sh --dataset /path/to/enhanced_frs_2023_24.h5
# Otherwise the committed results/*.json are used as-is.
# Stage 2: rebuild all figures + CSVs in results/figures/.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${RUN_PIPELINE:-0}" == "1" ]]; then
    uv run --frozen --extra simulation python analysis/run_all.py "$@"
fi

uv run --frozen python analysis/figures.py
uv run --frozen python analysis/fig_joint_matrix.py
echo "Done."
