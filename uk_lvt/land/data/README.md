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
