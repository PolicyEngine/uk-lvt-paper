# Figures

`analysis/figures.py` draws `fig1` to `fig8` from the committed
`results/lvt_results.json`, and `analysis/fig_joint_matrix.py` draws
`fig_joint_income_wealth`. Each figure is a PNG with a sibling CSV of the
plotted data. `analysis/regenerate_all.sh` runs both scripts.

## Superseded: `fig9_net_change_by_region` (not in the paper)

`fig9_net_change_by_region.png` and `fig9_net_change_by_region.csv` are
**superseded** and are not part of the paper's results. They show the average
net income change by region at the budget-neutral rate. PR #11 (merged
29 July 2026) removed the chart and its paragraph from the paper.

Unlike the constituency map in `results/geo/`, the chart is not mixed-vintage.
It was last drawn in commit `8c89069` (PR #10), from the same
`results/lvt_results.json` the paper uses. The CSV holds that file's
`impact_by_region` table, sorted by `avg_net_change`, and
`tests/test_figures_superseded.py` checks that it still does.

The files stay in the repository as an archive of the dropped figure.
`figures.py` does not redraw them by default. To redraw them from the
committed JSON, run `python analysis/figures.py --include-superseded`.
