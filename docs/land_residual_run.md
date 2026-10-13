# Residual land check: build report

Built a residual estimator for all 296 English authorities on
`land/residual-check`, based on latest observed 2018–April 2026 Price Paid
sales, LA HPI uprating, EHS floor areas, open MIT TPA rebuild assumptions, and
VOA CTSOP 2026 type/proxy-band stock weights. Source hashes, four small derived
tables, full methods and comparison outputs are committed. Individual
addresses and raw transaction records are not committed.

| Headline | Result |
|---|---:|
| Dwelling stock | 25,945,040 |
| Latest sales | 5,778,353 |
| Negative residuals clipped | 921,458 (15.95%) |
| England dwelling-weighted share | 36.91% |
| England land/value ratio | 51.42% |
| Implied England land | £5.268tn |
| Implied England property value | £10.244tn |
| Spearman against TPA | 0.985 |
| Spearman against MHCLG | 0.463 |

Regional dwelling means: North East 20.10%; North West 28.54%; Yorkshire/Humber
27.85%; East Midlands 24.74%; West Midlands 29.27%; East of England 39.58%;
London 62.27%; South East 42.39%; South West 34.43%.

The newest explicit public [ONS dwelling-land figure located](https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/articles/improvingestimatesoflandunderlyingotherbuildingsandstructuresinthenationalbalancesheetuk/2022)
is **UK end-2021 £5.9tn**: the England April 2026 total is 89.28% of it.
Geography and dates differ, so this is not aligned validation. The inspected
latest preliminary balance sheet (10 June 2026) publishes total land without
a dwelling-land breakdown. Verified bands and arm’s-length status remain
unavailable; current-price rank bands and standard category-A records are
explicit proxies. Depreciation is not modelled. TPA shares inputs and applies
national calibration, so its high rank correlation is a differential check.

Results and all reference regional means:
[`land_residual_regions.csv`](../results/land/land_residual_regions.csv).
The 20 largest disagreements **per reference**:
[`land_residual_disagreements.csv`](../results/land/land_residual_disagreements.csv).
[Full method, sources, assumptions and sensitivities](land_residual_method.md).

## Commands and verification

The sandbox prevents writes to the caller’s linked Git metadata. Git commands
therefore use workspace-local metadata under `.cache/residual-git`, retaining
an independent `land/residual-check` branch from fetched
`origin/land/d1-prototype`. No caller files or Git metadata were changed.

```bash
git --git-dir=.cache/residual-git fetch origin land/d1-prototype
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cache/residual-runtime/bin/python -m uk_lvt.land.residual_derive
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cache/residual-runtime/bin/python -m uk_lvt.land.residual
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cache/residual-runtime/bin/python -m pytest tests/test_land_residual.py -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .cache/residual-runtime/bin/python -m pytest tests/test_land_residual_ingestion.py -q
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cache/residual-py/bin/python -m uk_lvt.land.residual --check
git --git-dir=.cache/residual-git diff --check
```

Final verification: **74 passed** in `tests/test_land_residual.py`, then
**11 passed** in `tests/test_land_residual_ingestion.py`; no final skips or
failures. `residual --check` passed across both dependency environments, and
`git diff --check` passed. All 296 authorities, manifest hashes, convexity,
bounds, monotonicity and committed-byte reproduction were checked.

Derivation used Python 3.13.9, pandas 2.3.3, NumPy 2.3.3 and DuckDB 1.4.1.
The output check also passed under pandas 3.0.6, NumPy 2.5.4 and DuckDB 1.5.6.
An independent stdlib `csv`/`math.fsum` reconstruction verified every authority,
all regional means, totals, clipping counts, Spearman correlations and both
20-authority disagreement lists.

Initial fixture runs caught CSV API/SQL alias issues; all four SQL stages
were subsequently fixture-checked before the successful derivation. An
initial arithmetic-file run preceded derived artifacts and failed only the
three artifact-dependent checks. The final checks above are authoritative.
Only the two new test files were run locally, each separately, in the
foreground. Whole-suite and multi-file runs were skipped under the shared-host
rule; no heavy-run lock was bypassed. Paper/simulation checks were not run.

Large yearly raw files, scratch databases and task-created environments are
removed after verification. Small source caches and the isolated Git metadata
remain ignored. The workspace artifact `residual-land.patch` retains all commits for portability.

Draft PR URL will be recorded after creation.
