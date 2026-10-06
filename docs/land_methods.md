# Local-authority land shares for England: methods note (prototype)

**Status: prototype for review.** This is the first step of the authority-level
land values that the constituency-level LVT analysis needs. The shares here are
**raw**: they are not yet scaled to an ONS anchor or cross-validated. Nothing in
this note or in `results/land/` feeds the paper, the PolicyEngine UK model or
any published number, and none of these figures should be cited.

Code: `uk_lvt/land/`. Outputs:

- `results/land/la_lambda_raw.csv`: one row per English local authority (296).
- `results/land/la_lambda_raw_summary.csv`: England and regional aggregates of
  every variant under three weightings.

Rebuild from the committed tables (no downloads):

```bash
uv run --extra dev python -m uk_lvt.land.lambda_raw          # rewrites results/land/ and the tables below
uv run --extra dev python -m uk_lvt.land.lambda_raw --check  # exit 1 if anything would change
```

Rebuild the committed tables from the raw inputs:

```bash
uv run --extra land python -m uk_lvt.land.fetch      # download pinned inputs to .cache/land and verify sha256
uv run --extra land python -m uk_lvt.land.derive     # rewrite uk_lvt/land/data/*.csv and derived_manifest.json
```

## What is computed

For each of the 296 English authorities:

- **(a) a stock price.** In this prototype it is the UK House Price Index (UK
  HPI) average price for the authority in October 2023. This is a proxy; see
  step 3.
- **(b) raw λ_LA = MHCLG plot land per new dwelling ÷ stock price.**

The target concept for the finished dataset is the land value of the existing
residential stock, as an arithmetic mean, at one valuation date, with its level
anchored to ONS estimates of land under dwellings. The prototype stops short of that
in three ways, each listed under "Not done yet":

1. The numerator is MHCLG's residual value of a serviced 1 ha development site
   divided by the homes it would hold. It is land per *new* dwelling, not land
   under the existing stock, and MHCLG states that "these land values do not
   represent the market value of land" (guidance para 5). Only its
   cross-sectional shape is meant to carry through.
2. The denominator is a standardised transaction price, not the arithmetic mean
   value of the stock.
3. The level is not anchored.

## Inputs

Every input is listed in `uk_lvt/land/sources.yaml` with URL, publisher,
edition, licence, retrieval time and sha256, including inputs not fetched yet.
This step reads three of them, through the committed tables in
`uk_lvt/land/data/` (each traced to its source hash in
`uk_lvt/land/data/derived_manifest.json`):

| Source id | What is used | Committed table |
|---|---|---|
| `mhclg_lve_2023` | MHCLG, *Land value estimates for policy appraisal 2023* (valuation date 1 October 2023): sheets "Residential (adj for floor)" and "Residential", all densities and price percentiles | `mhclg_lve2023_residential.csv` (long form, 8,850 rows) |
| `ukhpi_average_prices_2026_07` | HM Land Registry, UK HPI average prices, July 2026 release: `Average_Price` for English authorities at 2023-10-01 | `ukhpi_la_average_price.csv` (295 rows) |
| `voa_ctsop_2026_summary` | VOA, Council Tax stock of properties 2026, table CTSOP1.0 (counts at 31 March 2026) | `voa_ctsop1_2026_england_la.csv` (296 rows) |

MHCLG's guidance is *Land value estimates for policy appraisal: guidelines for
use* (GOV.UK, 16 March 2026); paragraph numbers below refer to it.

## Steps

### 1. Authorities and codes

- **Universe.** The 296 English authorities in CTSOP1.0 (ONS codes E06: 63,
  E07: 164, E08: 36, E09: 33). These are referred to as 2023 codes: the latest
  new billing authorities in the CTSOP notes date from 1 April 2023, and the
  295 English authorities in UK HPI carry the same codes. Each authority's
  region is the CTSOP1.0 region row it sits under.
- **MHCLG code crosswalk.** MHCLG publishes 295 authorities. Barnsley and
  Sheffield appear under their pre-April-2023 codes and are mapped
  E08000016 → E08000038 and E08000019 → E08000039 (`geography.MHCLG_TO_2023`).
  Their values are used unchanged (assumption A1). They carry
  `method_flag = computed_code_crosswalk`.
- **Isles of Scilly (E06000053).** Neither MHCLG nor UK HPI publishes it. It
  keeps its own row and takes Cornwall's (E06000052) MHCLG values, stock price
  and therefore every λ, with `method_flag = imputed_from_E06000052`
  (assumption A2). It has 1,170 properties in CTSOP1.0. It is kept as its own
  row rather than merged into Cornwall so that every later step can join on
  2023 codes.
- All three inputs are already on 2023 codes, so the mapping of pre-2023
  districts onto the 2023 unitaries is not needed in this step. It is needed in
  D1.7, where data keyed on 2021 codes is joined.

### 2. MHCLG plot land per new dwelling

`plot land per new dwelling = V / d`, where, in the central case:

- `V` is MHCLG's value in £ per hectare from the "Residential (adj for floor)"
  sheet, **medium** density scenario (MHCLG's central scenario, para 33), at
  the **median** (50th percentile) new-build price per m² (para 35);
- `d` is MHCLG's dwellings per hectare for the medium scenario.

Facts from the guidance that matter here: each site is a serviced 1 ha site
whose net developable area is 80% of the gross area, "excluding London"
(para 31); densities are floored at 30 and capped at 400 dwellings per hectare
(para 34); the floor-adjusted sheet applies a minimum "floor value" built from
typical section 106 affordable housing and ten times the agricultural value
(para 36). In the committed table the floor-adjusted value is never below the
unadjusted one and is above it in 115 of the 4,390 cells where both exist
(checked in `tests/test_land_data.py`).

### 3. Stock price (prototype proxy: UK HPI)

The plan's route (latest Price Paid sale per dwelling, uprated with the
authority's UK HPI to October 2023 and reweighted to the VOA stock by band ×
type) is **not built** in this prototype; the complete Price Paid file is about
5.5 GB. The stock price is instead the UK HPI `Average_Price` for the authority
at 2023-10-01, from the July 2026 release (so it includes any revisions up to
that release). Assumption A3.

What that number is, from *About the UK House Price Index* (GOV.UK, updated
23 July 2026): the standard average price is the geometric mean price in
January 2015, carried backwards and forwards by a hedonic, mix-adjusted index;
below region level it is a 3-month moving average (the October 2023 figure
averages the August, September and October estimates). It is therefore a
standardised transaction price, not the arithmetic mean value of the stock.
For any fixed set of prices a geometric mean cannot exceed the arithmetic mean,
so if the index's standard mix matched the stock, this proxy would understate
the stock price and overstate λ. The size of that effect, and of the mix
difference, is not measured here.

### 4. Raw λ

`λ = plot land per new dwelling ÷ stock price`. Every λ column is reported on
[0, 1]: a ratio above 1 would be capped at 1 and the variant named in
`capped_variants`. With the current inputs no authority is capped in any
variant.

### 5. Sensitivities

All variants change one thing from the central case. A density factor `f`
replaces `d` with `f × d`.

- **Density ±25%** (`density_plus25`, `density_minus25`). MHCLG does not say
  whether its densities are per gross or per net hectare. Under the net
  reading, a gross hectare outside London holds 0.8 × `d` homes (80% net
  developable area), so land per home and λ are 25% higher: `f = 0.8`,
  `density_plus25`. The guidance states the 80% for sites outside London only,
  so under that reading London's λ would not change; `density_plus25` applies
  the 25% to the 33 London boroughs too and is the mechanical upper bound
  there. `density_minus25` (`f = 4/3`, λ × 0.75) is the symmetric lower bound;
  no mechanism for it is identified in the guidance. The basis question is
  open (see below).
- **Price percentile** (`price_p25`, `price_p75`). MHCLG's value at the 25th
  and 75th new-build price percentiles instead of the median, with the same
  stock price.
- **No floor value** (`unadjusted`). The "Residential" sheet, where MHCLG
  reports no value for negative residuals. Cumberland has no value at the
  median there, so this variant covers 295 authorities.
- **Plot-to-stock adjustment: not done** (see "Not done yet").

<!-- BEGIN GENERATED: sensitivity_table -->
| Variant | Definition | England, dwelling-weighted | England, value-weighted | Authorities with a value | Capped at 1 |
|---|---|---:|---:|---:|---:|
| `central` | Floor-adjusted sheet, medium (central) density, median new-build price percentile | 0.334 | 0.354 | 296 | 0 |
| `density_minus25` | As central with 4/3 x published density (lambda x 0.75) | 0.250 | 0.265 | 296 | 0 |
| `density_plus25` | As central with 0.8 x published density (lambda x 1.25; the net-hectare reading outside London) | 0.417 | 0.442 | 296 | 0 |
| `price_p25` | As central at MHCLG's 25th new-build price percentile | 0.242 | 0.265 | 296 | 0 |
| `price_p75` | As central at MHCLG's 75th new-build price percentile | 0.430 | 0.453 | 296 | 0 |
| `unadjusted` | As central from the sheet without MHCLG's floor value; missing where MHCLG reports no value | 0.335 | 0.354 | 295 | 0 |
<!-- END GENERATED: sensitivity_table -->

### 6. Aggregation

Two weightings, both computed for England and each region, for every variant:

- **Dwelling-weighted** (the work plan's definition): `Σ nᵢ λᵢ / Σ nᵢ`, with
  `nᵢ` = all properties in CTSOP1.0 at 31 March 2026 (assumption A4).
- **Value-weighted**: weights `nᵢ × Pᵢ`. This equals
  `Σ nᵢ × plotᵢ / Σ nᵢ × Pᵢ`, total plot land over total stock value, which is
  the same ratio form as a national land ÷ (land + structures) share. If the
  ONS anchor in D1.5 is such a ratio, the value-weighted aggregate is the one
  that matches it; the choice belongs to D1.5.

Both are convex combinations of the authority values (non-negative weights
normalised to sum to 1), so each lies between the lowest and highest λ it
averages. An unweighted mean is also reported.

Central case, raw (not anchored):

<!-- BEGIN GENERATED: regional_table -->
| Area | Authorities | Dwelling-weighted | Value-weighted | Unweighted mean | Lowest | Highest |
|---|---:|---:|---:|---:|---:|---:|
| England | 296 | 0.334 | 0.354 | 0.330 | 0.077 | 0.702 |
| North East | 12 | 0.256 | 0.258 | 0.263 | 0.182 | 0.399 |
| North West | 35 | 0.267 | 0.273 | 0.248 | 0.087 | 0.445 |
| Yorkshire and The Humber | 15 | 0.325 | 0.326 | 0.312 | 0.163 | 0.442 |
| East Midlands | 35 | 0.251 | 0.251 | 0.240 | 0.079 | 0.420 |
| West Midlands | 30 | 0.361 | 0.360 | 0.345 | 0.230 | 0.496 |
| East of England | 45 | 0.333 | 0.345 | 0.335 | 0.077 | 0.497 |
| London | 33 | 0.387 | 0.400 | 0.391 | 0.183 | 0.702 |
| South East | 64 | 0.373 | 0.374 | 0.383 | 0.132 | 0.550 |
| South West | 27 | 0.371 | 0.369 | 0.367 | 0.252 | 0.472 |
<!-- END GENERATED: regional_table -->

Lowest and highest central raw λ:

<!-- BEGIN GENERATED: extremes_table -->
| Rank | Lowest raw lambda | Value | Highest raw lambda | Value |
|---:|---|---:|---|---:|
| 1 | Southend-on-Sea | 0.077 | Enfield | 0.702 |
| 2 | East Lindsey | 0.079 | City of London | 0.693 |
| 3 | Blackpool | 0.087 | Westminster | 0.589 |
| 4 | Boston | 0.093 | Hammersmith and Fulham | 0.566 |
| 5 | Cumberland | 0.098 | Gosport | 0.550 |
<!-- END GENERATED: extremes_table -->

The three tables above are generated by `python -m uk_lvt.land.lambda_raw`
from `results/land/`; `tests/test_land_data.py` fails if they drift.

## Assumptions

- **A1.** MHCLG's values for Barnsley (E08000016) and Sheffield (E08000019)
  apply unchanged to E08000038 and E08000039. Whether the recoding came with a
  boundary change, and how large, was not checked.
- **A2.** Isles of Scilly takes Cornwall's MHCLG values and UK HPI price.
- **A3.** The UK HPI October 2023 average price stands in for the stock price
  (step 3).
- **A4.** Dwelling weights are CTSOP1.0 counts at 31 March 2026, two and a half
  years after the valuation date.
- **A5.** MHCLG's "Medium" density column is the central scenario of para 33.
- **A6.** The central case uses the floor-adjusted sheet, as the plan
  specifies.
- **A7.** The central case reads densities as dwellings per gross hectare
  (`f = 1`).
- **A8.** The density sensitivity is symmetric ±25% on λ (`f = 0.8` and
  `f = 4/3`) for every authority, London included.
- **A9.** Regions are taken from CTSOP1.0's region rows.
- **A10.** λ above 1 would be capped at 1 and flagged (not triggered).

## Tests

- `tests/test_land_lambda.py` (Hypothesis): 0 ≤ λ ≤ 1 for valid inputs and
  λ equal to the uncapped ratio whenever that ratio is at most 1; the weighted
  aggregation is a convex combination (bounded by the extremes, equal to
  `Σ αᵢ λᵢ` with `αᵢ ≥ 0`, `Σ αᵢ = 1`, invariant to rescaling the weights and
  to reordering); λ is non-increasing in the density factor, with
  `density_plus25` = 1.25 × central and `density_minus25` = 0.75 × central
  whenever nothing is capped; and the authority-table builder keeps each
  authority exactly once for generated inputs.
- `tests/test_land_data.py`: all 296 authorities present exactly once, on 2023
  codes, with the crosswalk and Scilly flags; the results and the tables in
  this note rebuild byte for byte from the committed tables; every committed
  table's sha256 matches `derived_manifest.json`, whose source hashes match
  `sources.yaml`; MHCLG's floor-adjusted values are never below the unadjusted
  ones, rise with the price percentile, and densities lie in [30, 400] with
  low ≤ medium ≤ high. Where the raw inputs are cached, it also checks that
  `derive` reproduces the committed tables byte for byte (skipped otherwise,
  as in CI).
- `tests/test_land_fetch.py`: the manifest validates; a download is accepted
  only when its sha256 matches and is discarded otherwise; a mismatching cached
  file fails loudly; local copies are reused only by hash.

## Not done yet

1. **Stock price from Price Paid (rest of D1.3).** Latest sale per dwelling
   from `hmlr_price_paid_complete`, uprated with the authority's UK HPI to
   October 2023, reweighted to the VOA stock by band (CTSOP1.0) × type
   (CTSOP3.0); then bridges from that arithmetic stock mean to the UK HPI
   average (this prototype's proxy) and to the ONS median prices
   (`ons_housing_affordability_2025`, whose file URL is not yet identified).
2. **Plot-to-stock sensitivity (rest of D1.4).** The stock is older, has
   different plot densities and carries no hope value. Inputs are pinned or
   listed (Live Table 126, CTSOP3.0 and 4.0, EPC D3); no adjustment is
   computed, and the central case for it is still to be chosen.
3. **ONS anchor (D1.5).** Scale λ so the national aggregate equals all-sector
   land under dwellings ÷ (that land + dwellings, AN.111); the latest series is
   to be requested from ONS (`ons_land_under_dwellings_latest`), with ONS's
   published 2020 estimate as the fallback. Separately check the implied
   household land against the household sector (S.14) in the national balance
   sheet 2025 (`ons_nbs_2025`, pinned), and publish the bridge from total land
   to household land with general government land shown separately. The
   weighting question in step 6 is decided here.
4. **Cross-validation (D1.6).** Against Tax Policy Associates' `cells.json`
   (`tpa_lvt_model_cells`, pinned, MIT licence) and Progress and Property
   authority aggregates (`progress_and_property_la_aggregates`, not available),
   flagging authorities whose |log ratio| exceeds 0.4 against two or more
   references, with rank correlations and bounds stated in advance.
5. **Constituency mapping (D1.7).** λ for each 2024 constituency as the
   dwelling-weighted mean of the λ_LA it overlaps (output-area lookups to both
   geographies); seat Band D the same way; mapping of pre-2023 districts for
   data on 2021 codes; interpolation flags for Scotland, Wales and Northern
   Ireland, which this prototype does not cover.
6. **Publication (D1.8).** The final dataset (code, name, dwellings, mean
   value, plot land, λ, method flag, reference comparisons) and a methods note
   covering the HPI, MHCLG-plot and Wealth and Assets Survey bridges.
7. **Open question to MHCLG:** are the published densities per gross or per net
   hectare? Until answered the ±25% sensitivity stands.
8. **Inputs not fetched yet:** UK HPI by property type and new/old files,
   Price Paid, Live Table 125, ONS affordability 2025, Welsh and Scottish
   council tax by band (both landing pages refused automated requests), the
   ONS land-under-dwellings series and Progress and Property aggregates (both
   by request).
9. **Live re-download of pinned inputs.** The pinned copies were downloaded on
   6 October 2026 (UTC) and checked against the live URLs by HTTP HEAD only:
   the size matches for the seven whose server reports one, and for UK HPI
   and the ONS balance sheet the server ETag equals the pinned copy's MD5 and
   SHA-1 respectively.
   `python -m uk_lvt.land.fetch` has not yet been run against the live URLs
   from a clean cache.
