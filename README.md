# uk-lvt-paper

**Replacing council tax with a 0.79% flat land value tax raises the same £58.5bn, leaves 67% of households better off (63–76% across incidence variants), and shifts the burden sharply up the wealth distribution. Simulated household by household in PolicyEngine UK.**

```
WAS property values → regional land shares → ONS benchmark →
PolicyEngine microsimulation → revenue-neutral rate → distributional analysis
```

## Pipeline stages

### 1. Land values (policyengine-uk)

Household land value = property wealth × regional land share + directly
owned land, computed in policyengine-uk. Land shares (42% North East to 85%
London) come from the MHCLG residual method; Scotland and Wales are
interpolated from English regions. Corporate land (£2.06tn, the ONS National
Balance Sheet aggregate held flat from 2024) is allocated to households in
proportion to corporate wealth, so that component is pinned to its aggregate
by construction. Household land is *not* pinned: it comes out at £5.38tn,
107% of the un-uprated ONS 2024 benchmark of £5.04tn, an excess the paper
attributes mainly to property-price uprating in the microdata. Total base
£7.44tn.

### 2. Microsimulation (`uk_lvt/pipeline_direct.py`)

One baseline `policyengine_uk.Microsimulation` on the Enhanced FRS. Every
scenario is then derived in closed form: a household's change in net income
is the council tax it no longer pays (net of council tax reduction) minus
`rate × land`. The module re-runs the full model with
`gov.contrib.ubi_center.land_value_tax.rate` and
`gov.contrib.abolish_council_tax` at 0.5%, 1% and 2% and refuses to write
results unless every household's full-model change matches the closed form to
within 8 float32 spacings of its largest income, benefit or tax total, and
the poverty rates match to 0.01pp (`uk_lvt/validation.py`).

### 3. Revenue-neutral rate (`uk_lvt/validation.py`)

Rate = net council tax revenue / total land value = £58.5bn / £7,436bn =
**0.79%** (0.786% to three decimals).

### 4. Table builders (`uk_lvt/analysis.py`)

Pure numpy/pandas transforms: weighted means/medians, decile tables,
winner/loser shares, poverty and Gini changes, revenue-by-rate grids.

### 5. Secondary analyses (`analysis/`)

`extensions.py` (robustness, capitalisation, deferral), `deep_dive.py`
(effective rates, progressivity, tenure, bootstrap, dividend recycling,
HVCTS), `submission_extras.py` (pass-through curve, joint matrix,
Reynolds–Smolensky), `bootstrap_ci.py` and `lambda_mc.py`. Each loads one
baseline on the same dataset (`extensions.py` also runs one reform to check
the arithmetic) and writes its own JSON in `results/`.

### 6. Figures (`analysis/figures.py`, `analysis/fig_joint_matrix.py`)

Read the committed JSON and write eight PNG + CSV pairs plus the joint
income–wealth heatmap to `results/figures/`. No licensed data needed.

## Headline results

| Measure | Value |
| --- | --- |
| Budget-neutral LVT rate | 0.79% |
| Revenue replaced | £58.5bn |
| Poverty change (BHC, individuals) | −0.23pp |
| Poverty change (AHC, individuals) | −0.69pp |
| Income Gini change (equivalised, person-weighted) | −0.0025 (essentially unchanged) |
| Households gaining | 67.4% |
| Income decile 1 average change | +£584/yr |
| Income decile 10 average change | −£1,429/yr |
| Top wealth decile average change | −£5,871/yr |
| Share of land held by top wealth decile | 47.5% |
| Council tax as % of property value | 1.78% (under £150k) → 0.07% (over £2m) |
| Kakwani index vs wealth | council tax −0.523, LVT −0.032 |
| Kakwani index vs income | council tax −0.201, LVT −0.084 |
| Outright owners / renters | −£755 / +£235 to +£773 per year |

`tests/test_results_golden.py` pins the headline numbers to the committed
JSON, so a regeneration that moves them fails the tests.

## Reproduce

### Pins

| Input | Pin |
| --- | --- |
| Python | 3.13 (`.python-version`) |
| policyengine-uk | 2.89.4 |
| policyengine-core | 3.30.3 |
| Everything else | `uv.lock` (numpy 2.5.3, pandas 3.0.6, …) |
| Dataset | `enhanced_frs_2023_24.h5` at revision `1.56.14` of the Hugging Face repository `policyengine/policyengine-uk-data-private`, SHA-256 `584ae33d80ca0431254610a3f8254d132da73477d31966d6446282861ecae50d` |
| ONS land targets | three files from policyengine-uk-data at commit `ebf733c6` (tag `1.56.14`), fetched by `scripts/fetch_uk_data_targets.py` |

The dataset file at revision `1.56.14` was uploaded with data release 1.55.10
and is not listed in the 1.56.14 release manifest (which covers the 2024–25
files only); the revision tag and the SHA-256 identify it exactly. The July
2026 results did not record their policyengine-core version.
policyengine-core 3.30.3 was the newest release on PyPI when
`results/lvt_results.json` was last regenerated (27–28 July 2026). Every
committed JSON reproduces exactly under it; `lvt_results.json` also
reproduces exactly under 3.32.18.

Every results JSON carries a `provenance` block with the policyengine-uk,
policyengine-core, numpy and pandas versions, the dataset source and the
SHA-256 of the dataset file actually read. The tests check that these match
`pyproject.toml`, `uv.lock` and the dataset pin.

### Without licensed data

```bash
uv sync --frozen --extra dev
uv run --frozen pytest
uv run --frozen python analysis/figures.py           # figures from the committed JSON
uv run --frozen python analysis/fig_joint_matrix.py
cp paper/main.pdf /tmp/committed.pdf
(cd paper && tectonic main.tex)
bash scripts/check_paper.sh paper/main.pdf /tmp/committed.pdf
```

The figure CSVs reproduce byte for byte. The PNGs are rewritten with
different bytes, so restore them with `git checkout results/figures` unless
a figure has changed on purpose.

### With licensed data

```bash
uv sync --frozen --extra simulation
UK_DATA_ROOT=$(uv run --frozen python scripts/fetch_uk_data_targets.py)
export HF_TOKEN=...   # a Hugging Face token whose account has dataset access
uv run --frozen python -m uk_lvt.pipeline_direct --uk-data-root "$UK_DATA_ROOT"
uv run --frozen python analysis/run_all.py --uk-data-root "$UK_DATA_ROOT" --with-resampling
git diff --stat results/   # expect no changes
```

`run_all.py` reruns `pipeline_direct` and then the five secondary scripts.
To read a local copy of the dataset instead of downloading it, pass
`--dataset /path/to/enhanced_frs_2023_24.h5` to `pipeline_direct` or
`run_all.py`, or set `UK_LVT_DATASET` for the individual scripts. The JSONs
then differ from the committed ones only in the `dataset` fields, and
`provenance.dataset_sha256` shows whether the copy is the pinned file.

Set `HF_TOKEN`, not `HUGGING_FACE_TOKEN`: the dataset repository is public
but gated, and policyengine-core 3.30.3 passes `HUGGING_FACE_TOKEN` only to
repositories that report themselves as private, so the download relies on
huggingface_hub reading `HF_TOKEN`.

`uk-lvt-build` (the console script) and `analysis/regenerate_all.sh` with
`RUN_PIPELINE=1` drive the same canonical path. `uk_lvt/pipeline.py`, the
older policyengine.py v4 client pipeline, is kept only as a cross-check: its
poverty and Gini conventions differ from the paper's, no script runs it
(`pipeline_direct` reuses only its ONS target loader), and the client it
needs is not in the locked environment.

## Microdata and the UK Data Service licence

The Enhanced FRS contains individual-level Family Resources Survey microdata,
which PolicyEngine uses under UK Data Service licence terms. The file sits in
a gated Hugging Face repository; PolicyEngine grants access to Hugging Face
accounts whose holders can show licensed access to the underlying UK Data
Service data (see the [policyengine-uk getting-started guide](https://github.com/PolicyEngine/policyengine-uk/blob/main/docs/book/usage/getting-started.md)).

This repository is public. It holds code and aggregate outputs only
(`results/*.json`, figures, the paper). Never commit the dataset, a copy of
it, or any household-level extract or intermediate file derived from it.

## Superseded artefacts

- `results/geo/` holds a constituency map and CSV dropped from the paper in
  PR #12: they were produced on a different microdata build and are
  mixed-vintage. See `results/geo/README.md`. `analysis/geo_map.py`, which
  wrote them, refuses to run without `--allow-superseded`.
- The regional bar chart that `figures.py` still drew as `fig9` left the
  paper in PR #11. Its generator and outputs are removed; the regional numbers
  remain in `results/lvt_results.json` (`impact_by_region`).

## Known limitations

- Static microsimulation: no behavioural response and no capitalisation in
  the central results (the paper reports a capitalisation appendix).
- Land shares are regional averages applied to WAS property values, not
  property-level land valuations; Scotland and Wales are interpolated from
  English regions.
- Corporate land is allocated to households in proportion to corporate
  wealth, ignoring foreign ownership — a strong incidence assumption, tested
  in the robustness table.
- Results use policyengine-uk 2.89.4 and the Enhanced FRS 2023–24 file pinned
  above.
- Poverty is reported as the share of *individuals* (HBAI convention) and the
  income Gini is person-weighted over equivalised income; winners/losers use
  a ±£1 dead-band.
- The model's own `household_wealth_decile` is degenerate (decile 1 empty,
  22.6% of households in decile 2); wealth deciles here are reconstructed as
  equal-weight deciles.

## References

- Blog post: [How replacing council tax with a land value tax would affect UK households](https://progressandpoverty.substack.com/p/how-replacing-council-tax-with-a) (Progress and Poverty)
- ONS, UK National Balance Sheet estimates (land values)
- [PolicyEngine UK](https://policyengine.org/uk)
