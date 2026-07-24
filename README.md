# uk-lvt-paper

**Replacing council tax with a 0.77% flat land value tax raises the same £57.6bn, leaves 68% of households better off, and shifts the burden sharply up the wealth distribution — while barely moving income-based inequality or poverty statistics. Simulated household by household in PolicyEngine UK.**

```
WAS property values → regional land shares → ONS calibration →
PolicyEngine microsimulation → revenue-neutral rate → distributional analysis
```

## Pipeline stages

### 1. Land values (`uk_lvt/pipeline.py`)

Household land value = WAS property value × regional land share + directly
owned land, computed in policyengine-uk. Land shares (42% North East to 85%
London) come from the MHCLG residual method; Scotland and Wales are
interpolated from English regions. Corporate land (£2.06tn, the ONS National
Balance Sheet aggregate held flat from 2024) is allocated to households in
proportion to corporate wealth, so that component is pinned to its aggregate
by construction. Household land is *not* pinned: it comes out at £5.40tn,
107% of the un-uprated ONS 2024 benchmark, the excess being the property-price
uprating embodied in the microdata. Total base £7.46tn.

### 2. Microsimulation (`uk_lvt/pipeline.py`)

Baseline and reform runs on the Enhanced FRS through the policyengine.py v4
client. Reforms set `gov.contrib.ubi_center.land_value_tax.{rate,
household_rate, corporate_rate}` and `gov.contrib.abolish_council_tax`.

### 3. Revenue-neutral rate (`uk_lvt/pipeline.py`)

Rate = net council tax revenue / total land value = £57.6bn / £7,463bn =
**0.77%**.

### 4. Table builders (`uk_lvt/analysis.py`)

Pure numpy/pandas transforms: weighted means/medians, decile tables,
winner/loser shares, poverty and Gini changes, revenue-by-rate grids.

### 5. Figures (`analysis/figures.py`)

Reads the committed `results/lvt_results.json` and writes eight PNG + CSV
pairs to `results/figures/` — no licensed data needed.

## Headline results

| Measure | Value |
| --- | --- |
| Budget-neutral LVT rate | 0.77% |
| Revenue replaced | £57.6bn |
| Poverty change (BHC) | +0.04pp (unchanged) |
| Poverty change (AHC) | −0.70pp |
| Income Gini change | +0.0003 (unchanged) |
| Households gaining | 68.2% |
| Income decile 1 average change | +£647/yr |
| Income decile 10 average change | −£963/yr |
| Top wealth decile average change | −£5,752/yr |
| Share of land held by top wealth decile | 47.4% |

## Reproduce

```bash
pip install -e ".[dev]"
pytest
python analysis/figures.py   # rebuilds all figures from results/lvt_results.json
```

The full simulation pipeline additionally needs `pip install -e
".[simulation]"`, a Hugging Face token with access to the licensed Enhanced
FRS dataset, and policyengine-uk-data for the ONS land targets:

```bash
python -m uk_lvt.pipeline_direct   # regenerates results/lvt_results.json
python analysis/extensions.py      # robustness, capitalisation, deferral
```

`pipeline_direct` runs one baseline simulation and derives every scenario in
closed form, re-running the full model at three rates to verify that the
arithmetic matches (it refuses to write results if it does not). The older
`uk-lvt-build` entry point drives the policyengine.py v4 client and runs one
solver pass per scenario.

## Known limitations

- Static microsimulation: no behavioural response and no capitalisation in
  the central results (the paper reports a capitalisation appendix).
- Land shares are regional averages applied to WAS property values, not
  property-level land valuations; Scotland and Wales are interpolated from
  English regions.
- Corporate land is allocated to households in proportion to corporate
  wealth, ignoring foreign ownership — a strong incidence assumption, tested
  in the robustness table.
- Poverty *levels* are sensitive to the policyengine-uk version. Results here
  are pinned to 2.88.20; an earlier release gave a −0.65pp BHC poverty change
  where this one gives +0.04pp. Revenue, land and winner-share results are
  not affected.
- The model's own `household_wealth_decile` is degenerate (decile 1 empty,
  22.6% of households in decile 2); wealth deciles here are reconstructed as
  equal-weight deciles.

## References

- Blog post: [How replacing council tax with a land value tax would affect UK households](https://progressandpoverty.substack.com/p/how-replacing-council-tax-with-a) (Progress and Poverty)
- ONS, UK National Balance Sheet estimates (land values)
- [PolicyEngine UK](https://policyengine.org/uk)
