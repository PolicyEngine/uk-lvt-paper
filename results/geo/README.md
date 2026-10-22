# Superseded: mixed-vintage constituency map (do not cite)

The two files in this folder are **superseded** and are not part of the
paper's results:

- `constituency_income_change.csv`: average household net income change by
  2024 Westminster constituency;
- `map_income_change.png`: the choropleth drawn from it.

`analysis/geo_map.py` produced them, last regenerated in commit `f42b30b`
(27 July 2026). They mix data vintages: the script ran on a separate local
`enhanced_frs_2024_25.h5` build whose 52,576 households matched a local
constituency weight matrix, not on the pinned `enhanced_frs_2023_24.h5`
(53,508 households) behind every other number in the paper, and it re-solved
the budget-neutral rate on that dataset (see the comments at the top of
`analysis/geo_map.py`). PR #12 (merged 31 July 2026) dropped the map and its
prose from the paper for this reason. The paper now defers constituency
results to a companion exercise on same-vintage weights
(`paper/sections/results.tex`, geography section).

The files stay in the repository so that the history of the dropped figure
is visible. They cannot be regenerated from this repository: `geo_map.py`
reads absolute paths on one developer's machine, and it now refuses to run
unless passed `--allow-superseded`.
