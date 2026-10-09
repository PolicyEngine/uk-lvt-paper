# Committed derived tables

Small tables extracted from public inputs by `python -m uk_lvt.land.derive`.
Do not edit them by hand: `derived_manifest.json` records each file's sha256
and the sha256 of the raw source it came from (as pinned in
`../sources.yaml`), and `tests/test_land_data.py` fails if they disagree.

| File | Source (id in `sources.yaml`) |
|---|---|
| `mhclg_lve2023_residential.csv` | MHCLG, Land value estimates for policy appraisal 2023 (`mhclg_lve_2023`) |
| `ukhpi_la_average_price.csv` | HM Land Registry, UK House Price Index, July 2026 release (`ukhpi_average_prices_2026_07`) |
| `voa_ctsop1_2026_england_la.csv` | Valuation Office Agency, Council Tax stock of properties 2026, CTSOP1.0 (`voa_ctsop_2026_summary`) |

## Attribution

Contains public sector information licensed under the Open Government Licence
v3.0 (https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/):
data from the Ministry of Housing, Communities and Local Government, HM Land
Registry and the Valuation Office Agency. Crown copyright and database right.

### UK House Price Index

`ukhpi_la_average_price.csv`, and the `stock_price` column of
`results/land/la_lambda_raw.csv` built from it, republish UK HPI data. HM Land
Registry requires this statement when UK HPI data is republished
([About the UK House Price Index, section 7, "Republishing data"](https://www.gov.uk/government/publications/about-the-uk-house-price-index/about-the-uk-house-price-index#republishing-data)):

> Contains HM Land Registry data © Crown copyright and database right 2020. This data is licensed under the Open Government Licence v3.0.

The same section asks for the nature of the data and the dates it covers.
The table holds the UK HPI standard average price (`Average_Price`) of each
English local authority for October 2023 (`date` 2023-10-01), from the July
2026 data release (published 16 September 2026, downloaded 6 October 2026).
It is not an average of October 2023 sales: the UK HPI takes the geometric
mean price in January 2015 and moves it back and forward with the index, and
below region level it publishes a 3-month moving average (here August,
September and October 2023).
