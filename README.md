# uk-lvt-paper

**Replacing council tax with a 0.77% flat land value tax leaves the average household in the bottom eight income deciles better off, lifts households out of poverty, and raises the same £57.6bn — a revenue-neutral swap simulated household by household in PolicyEngine UK.**

```
WAS property values → regional land shares → ONS calibration →
PolicyEngine microsimulation → revenue-neutral rate → distributional analysis
```

## Pipeline stages

### 1. Land values (`uk_lvt/pipeline.py`)

Household land value = WAS property value × regional land share, imputed
upstream in policyengine-uk-data. Corporate land (£2.06tn) is allocated to
households in proportion to corporate wealth. Totals are calibrated to the
ONS National Balance Sheet: £7.46tn in 2026–27, uprated from £7.10tn (2024)
by OBR per-capita nominal GDP growth.

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
| Poverty change (BHC) | −0.65pp |
| Poverty change (AHC) | −1.04pp |
| Income Gini change | +0.0004 |
| Households gaining | 68% |
| Deciles better off on average | 1–8 |
| Decile 1 average change | +£481/yr |
| Decile 9 average change | −£991/yr |

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
uk-lvt-build --year 2026          # or: python analysis/run_all.py
```

## Known limitations

- Static microsimulation: no behavioural response, no house-price or rent
  capitalisation effects.
- Land shares are regional averages applied to WAS property values, not
  property-level land valuations.
- Corporate land is allocated to households in proportion to corporate
  wealth, a strong incidence assumption.

## References

- Blog post: [How replacing council tax with a land value tax would affect UK households](https://progressandpoverty.substack.com/p/how-replacing-council-tax-with-a) (Progress and Poverty)
- ONS, UK National Balance Sheet estimates (land values)
- [PolicyEngine UK](https://policyengine.org/uk)
