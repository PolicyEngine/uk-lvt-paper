# Transaction residual land shares for England (prototype)

This independent check uses Price Paid transactions, EHS floor areas and open
TPA rebuild assumptions. It does not use MHCLG development-site values in its
construction. It is provisional: transaction coverage, imputed council-tax
bands and replacement costs remain substantive assumptions. Nothing here
feeds the paper or PolicyEngine production estimates.

Code: `uk_lvt/land/residual*.py`. Outputs are the 296-row
`results/land/la_lambda_residual.csv`, regional means, 20 largest absolute
share disagreements **per reference**, and `land_residual_summary.json`.
Address-free source cells are committed under `uk_lvt/land/data/`; raw data and
scratch files are kept in git-ignored `.cache/land/`.

## Sources and reproducibility

Every external input is pinned in `uk_lvt/land/sources.yaml` with URL, licence,
retrieval timestamp, byte size and SHA256. `data/residual_manifest.json` links
all derived tables and existing prototype inputs to their hashes. Raw
sources are verified before derivation. Offline result rebuilding always
verifies the derived tables and source pins, and verifies any raw sources
still cached. Missing raw files are permitted for the small-table rebuild.

| Input | Use |
|---|---|
| HM Land Registry Price Paid yearly CSVs, 2018–2026, October 2026 snapshot | Individual sale prices; yearly files only |
| UK HPI July 2026 release, average prices monthly series | LA uprating to April 2026 |
| VOA CTSOP 2026, CTSOP1.0 and CTSOP3.0 | 296-authority universe, regions, stock totals and band/type weights at 31 March 2026 |
| [EHS 2018 Size of English homes annex tables](https://www.gov.uk/government/statistics/english-housing-survey-2018-size-of-english-homes-fact-sheet), published 9 July 2020, AT1.2 and AT1.5 | Regional/type floor-area fallback |
| [TPA preprocessing/config.py](https://github.com/DanNeidle/lvt_model_2026/blob/fef94f9e7a36ada25a9c29f1f6827759a12eed07/preprocessing/config.py), commit fef94f9 | Open rebuild constants; MIT |
| TPA webapp/cells.json and preprocessing stages 1/2 at the same commit | Comparison and verification of its construction |
| [ONS preliminary national balance sheet, latest public edition checked 13 October 2026](https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/datasets/preliminaryuknationalbalancesheetestimates), released 10 June 2026, and latest NBS CSV | Establish what the latest edition actually publishes |
| [ONS 2022 land article, Figure 1](https://www.ons.gov.uk/economy/nationalaccounts/uksectoraccounts/articles/improvingestimatesoflandunderlyingotherbuildingsandstructuresinthenationalbalancesheetuk/2022) | Historical UK end-2021 dwelling-land fallback |
| Prototype committed MHCLG-based shares | Comparison only |

Contains HM Land Registry data © Crown copyright and database right 2021.
This data is licensed under the Open Government Licence v3.0. The Price Paid
address fields have additional third-party rights described on the
[yearly download page](https://www.gov.uk/government/statistical-data-sets/price-paid-data-yearly-file).
Only aggregate results are redistributed; no addresses are committed.
For UK HPI: Contains HM Land Registry data © Crown copyright and database
right 2020. This data is licensed under the Open Government Licence v3.0.

TPA assumptions are attributed to Tax Policy Associates, © 2026, under the MIT
licence reproduced in `data/TPA_LICENSE.txt`. Its cost constants are read by
AST literal evaluation; none of its source code is executed. No BCIS product
or licensed database is used. The regional factors are assumptions supplied
openly by TPA; its comments describe their BCIS/ABI-style origin.

Rebuild results without downloading:

```bash
OMP_NUM_THREADS=4 uv run --extra land --extra dev python -m uk_lvt.land.residual
OMP_NUM_THREADS=4 uv run --extra land --extra dev python -m uk_lvt.land.residual --check
```

To rebuild raw-derived cells, fetch only the source IDs listed in
`residual_manifest.json` (or use `--seed` with copies matching the pins), then
run `python -m uk_lvt.land.residual_derive`. For example, expand the complete
ID list from the committed provenance file without fetching unrelated inputs:

```bash
uv run --extra land python - <<'PYTHON'
import json, subprocess, sys
with open("uk_lvt/land/data/residual_manifest.json") as f:
    ids = list(json.load(f)["sources"])
subprocess.run([sys.executable, "-m", "uk_lvt.land.fetch", "--only", *ids], check=True)
PYTHON
OMP_NUM_THREADS=4 uv run --extra land python -m uk_lvt.land.residual_derive
```

 Do not fetch the complete Price
Paid source. `residual_derive` uses two DuckDB threads, a 512 MB memory limit,
and deletes its scratch database. Monthly refreshed yearly files can change;
the fetcher intentionally rejects changed bytes. Preserve pinned source
copies externally to reproduce that exact vintage after a publisher revision.

## Transactions, geography and price

The observation window is 1 January 2018 to 30 April 2026. The latest eligible
sale **within that window**, rather than the complete 1995-onward history, is
kept per property. Homes not sold in this window are represented only through
stock weights. Deeds later than April 2026 are excluded even though the source
snapshot includes later registrations. The valuation date `2026-04-01` denotes
the April monthly HPI observation, not a daily price estimate.

Eligibility requires category A, property type D/S/T/F, record status A or C,
positive price and a nonempty postcode and primary address. Category A is
standard single-property residential Price Paid data and serves as a proxy
for ordinary market sales. It is **not a verified arm’s-length indicator**:
nonmarket discounts, shared ownership and unusual transactions cannot all be
identified from these fields. Deleted and additional category B records are
excluded. Equal-date ties are resolved deterministically by transaction ID.

Properties are identified by postcode + primary address + secondary address
+ street. The secondary address distinguishes flats. Address changes can
split one dwelling; shared or missing unit identifiers can collapse distinct
dwellings. No claim of UPRN-quality linkage is made. Transaction IDs identify
sales rather than properties. No individual identifiers leave the cache.

Prices use the **district** field, not the postal town or county. Exact names
and aliases map to the prototype’s April 2023 codes. Explicit predecessor
lists combine the 2019 Dorset/BCP and Suffolk changes, Buckinghamshire,
Northamptonshire, and the 2023 Cumbria, North Yorkshire and Somerset changes.
Barnsley and Sheffield use the prototype’s new codes. York, Milton Keynes,
Bath and North East Somerset and North Somerset stay separate. District
crosswalking cannot resolve any within-district boundary transfer or erroneous
address geography. All unmatched districts and record counts are reported.

Each price is multiplied by that authority’s April 2026 UK HPI average price
divided by its sale-month average price. This ratio carries the LA index
change; it does not replace the transaction level with the HPI’s geometric
mean. Scilly retains its own sales and takes Cornwall’s index, as in the
prototype. Missing HPI joins fail rather than silently substituting another
geography. HPI is mix adjusted and the three-month moving average below region level remains a limitation;
individual appreciation, extensions and condition changes are not observed.

## Stock weights and proxy bands

VOA CTSOP3.0 supplies joint band/type counts. D/S/T/F map to houses and flats;
bungalows are distributed among D/S/T in proportion to known house types in
the same LA/band, falling back to the LA mix. Annexes, mobile homes and unknown
types are distributed across the known types. Suppressed nonzero counts below
five use 2.5. A common LA factor reconciles rounding to CTSOP1.0’s total. The
weights are fractional estimates, not exact population counts.

Price Paid has no council-tax band, and no property-level verified band match
has been obtained. Sales are ranked by uprated price within the **whole LA**;
midpoint ranks are sliced at the cumulative stock band shares. This monotone
current-price proxy follows TPA’s published approach, with joint type/band
reweighting added here. Current ranks need not equal April 1991 band ranks.
It must not be interpreted as an observed council-tax-band assignment.

For each LA/type/proxy-band cell, each observed sale represents stock divided
by the number of observed sales. Unobserved cells use that LA/type’s
sale-count-weighted mean; if the entire type lacks sales, they use the region’s
type mean. These cells have zero observed sales and are flagged. The stock
fraction using a fallback is published. A type-only weighting sensitivity
removes the proxy bands; rebuild-cost sensitivities multiply all costs by
0.75 and 1.25 and recompute clipping before averaging.

## Floor areas, rebuild costs and aggregation

`agent-secret list` found no EPC key; `agent-secret get EPC_API_KEY` also
returned no stored item. Anonymous EPC requests did not return certificate
data. The official [API guidance](https://get-energy-performance-data.communities.gov.uk/guidance/energy-certificate-data-apis)
requires GOV.UK One Login. No registration or credential entry was attempted.

EHS AT1.5 provides regional house/bungalow and flat means. For D/S/T, the
national detailed type mean from AT1.2 is multiplied by the region’s
house/bungalow mean divided by England’s house/bungalow mean. These detailed
regional areas are an explicit separability assumption, **not a published
joint D/S/T-by-region table**. F uses the directly published regional flat
mean. National D/S/T means are approximately 149.154, 97.258 and 87.817 m².
The England house/bungalow mean is 102.848 m². EHS covers 2018 non-bedsit
stock and cannot measure each sale’s area or subsequent extensions.

TPA base costs are £2,100/m² D, £1,950 S, £1,900 T and £2,400 F. Region factors
are 0.92 North East, 0.95 North West, 0.93 Yorkshire/Humber, 0.96 East Midlands,
0.97 West Midlands, 1.02 East of England, 1.15 London, 1.05 South East and
1.00 South West. No national calibration, cost scaling κ, or price scaling
is applied to our estimate. These are the assumptions in TPA’s model valued April 2026; no separately measured cost vintage or inflation series is supplied.

For each observed dwelling, `land = max(0, price − area × cost)` and
`share = land / price`. Negative residuals are clipped **before** aggregation;
the number clipped counts observed latest sales, never imputed cells. Within
an LA the reported main share is the stock-weighted arithmetic mean of these
individual ratios. It is a convex combination of cell means and, within
observed cells, individual shares. Its bounds are [0,1] and it cannot rise
when rebuild costs rise. It differs from the ratio of total land to total
property value, also supplied as `lambda_value_weighted`.

The implied England residential land total sums stock × mean clipped land
in each cell. Multiplying dwelling-weighted mean shares by mean property
prices would give a different, generally incorrect total. Regional means of
LA shares use VOA dwelling counts. National and regional ratios of total
land to total value are reported separately.

Replacement costs assume new structures. **Depreciation is not modelled.**
Older or poorly maintained structures may be worth less than replacement
cost, so this can understate residual land, especially in lower-price areas.
Bungalows receive D/S/T house areas despite typically smaller EHS areas;
this can also understate land. Conversely, fixed type averages understate
areas of unusually large homes and overstate their residual land. Area,
quality, age, plot size, location and price are jointly distributed; a
regional mean cannot preserve those relationships or nonlinear clipping.

Flats and leaseholds are retained. A flat’s sale price measures the sold
interest, which may include lease length, ground-rent and service-charge
terms. Its attributed structure cost is a type-average unit cost, not a
surveyed share of the entire block. Freeholder land interests, communal
land, conversion costs, short leases and enfranchisement are not separately
valued. Summing leasehold residuals need not recover all freehold land under
blocks. Bungalow classification and mobile-home allocation are additional
stock/type approximations.

## Comparisons and national scope

TPA cells contain authority/band stock counts and value/land decile means.
Its reference authority share is total stock-weighted land divided by total
stock-weighted value, not an average of band ratios. It uses price-rank bands,
a different area construction, no type-stock reweighting, portfolio/outlier
cleaning, and national calibration to approximately £8 trillion of housing
and a 55% ratio of land to property totals in its retained sold-home sample (its resulting stock ratio is 55.39%). Its geography is useful as a differential check; its
level is not an independent national benchmark. We report rank correlations
for our main dwelling-weighted share and, separately, our comparable
value-weighted ratio. TPA shares Price Paid, HPI and the rebuild constants
with this estimator, so correlation is not a wholly independent validation.
Its stock counts are from CTSOP 2025; regional comparisons here use the common
CTSOP 2026 dwelling weights. The prototype MHCLG share is valued October 2023;
our April 2026 estimate differs in both construction and valuation date.

The latest inspected ONS preliminary edition (10 June 2026) and current NBS
CSV publish AN.211 **total land**, without the land-under-dwellings subclass.
Its UK 2025 total land is £6.872455 trillion, which is not residential land.
The latest explicit public dwelling-land value found is £5.9 trillion for
UK end-2021, rounded to £0.1 trillion, in the June 2022 article. Our England
April 2026/31 March 2026 stock total is not geographically or temporally aligned
with that UK value. Their ratio is descriptive, not an estimated shortfall,
scaling factor or validation result. A latest, UK/England-aligned dwelling-land
comparison remains unavailable; no data request was sent.

## Results

All 296 authorities appear once, representing 25,945,040 dwellings. From
8,460,348 raw rows, 5,778,353 latest standard sales are retained; 921,458
(15.95%) have negative residuals clipped to zero.
The unmatched district names are exclusively the 22 Welsh authorities.
Leasehold sales: 1,295,781; new-build sales: 640,383.

The England dwelling-weighted mean is **36.91%**, versus **33.37%** for MHCLG.
The ratio of total land to total property value is **51.42%**. These answer
different questions and should not be substituted for each other.

Spearman across 296 LAs is **0.985 against TPA** and
**0.463 against MHCLG**. Using our value-weighted LA ratios
instead gives **0.982** and
**0.439**, respectively. This indicates
strong agreement with TPA's transaction residual ranking and materially
weaker agreement with the development-site ranking; shared inputs prevent
treating the TPA correlation as wholly independent corroboration.

The implied England land total is **£5.268 trillion**,
and implied property value is **£10.244 trillion**.
England April 2026 land is 89.28% of the historical
UK end-2021 £5.9 trillion dwelling-land reference. This is not an aligned ONS
validation or an estimate of an England shortfall. The transaction-based
property total is also substantially above TPA's calibrated £8.001 trillion,
which reinforces the sales-versus-stock and level uncertainties.

Only 0.0337% of stock uses sparse-cell fallbacks. Costs −25% and +25%
give England dwelling means of 50.59% and 25.91%, respectively.
The largest main-versus-type-only difference is
7.93 percentage points
in Preston, illustrating sensitivity to proxy band weighting.

| Area | Residual, dwelling-weighted | Residual, value-weighted | TPA LA ratios, dwelling-weighted | MHCLG, dwelling-weighted |
|---|---:|---:|---:|---:|
| England | 36.91% | 51.42% | 48.67% | 33.37% |
| East Midlands | 24.74% | 32.58% | 35.60% | 25.14% |
| East of England | 39.58% | 48.55% | 50.70% | 33.32% |
| London | 62.27% | 72.81% | 72.76% | 38.68% |
| North East | 20.10% | 32.25% | 29.00% | 25.57% |
| North West | 28.54% | 40.21% | 40.28% | 26.66% |
| South East | 42.39% | 51.69% | 56.91% | 37.34% |
| South West | 34.43% | 43.49% | 49.51% | 37.05% |
| West Midlands | 29.27% | 38.51% | 40.35% | 36.13% |
| Yorkshire and The Humber | 27.85% | 39.26% | 37.15% | 32.46% |


The 20 largest absolute share differences **for each** reference are in
[`land_residual_disagreements.csv`](../results/land/land_residual_disagreements.csv).
Their unit is a share difference (multiply by 100 for percentage points).

Largest five TPA disagreements:

| Authority | Residual | Reference | Difference, percentage points |
|---|---:|---:|---:|
| Portsmouth | 28.56% | 47.86% | -19.29 |
| Eastbourne | 28.42% | 47.68% | -19.26 |
| Test Valley | 37.43% | 56.03% | -18.60 |
| Gosport | 22.49% | 40.62% | -18.13 |
| Pendle | 15.42% | 33.46% | -18.04 |

Largest five MHCLG disagreements:

| Authority | Residual | Reference | Difference, percentage points |
|---|---:|---:|---:|
| Haringey | 67.91% | 27.94% | +39.97 |
| Richmond upon Thames | 71.82% | 32.33% | +39.49 |
| Kensington and Chelsea | 82.42% | 43.36% | +39.06 |
| Redbridge | 60.09% | 21.64% | +38.45 |
| Southend-on-Sea | 46.02% | 7.71% | +38.31 |


## Validation and remaining work

Property tests cover bounded shares, convex aggregation and monotone rebuild
costs. Fixture tests cover latest-sale selection, eligibility, address/unit
keys, district geography, HPI joins, stock imputation, price-rank bands and
sparse-cell donor rules. Data checks require all 296 authorities exactly once,
matching manifest hashes and byte-for-byte offline result reproduction.

Remaining work includes verified council-tax bands and property identifiers;
EPC or other property-specific areas; depreciation, bungalow and leasehold
refinements; a longer transaction history and sales-versus-never-sold stock
validation; stronger nonmarket-sale identification; and a latest aligned ONS
dwelling-land subclass. No ONS anchoring is applied here.

Final local verification: `pytest tests/test_land_residual.py -q` — 74 passed;
`pytest tests/test_land_residual_ingestion.py -q` — 11 passed. Each file ran
separately in the foreground with at most four OMP threads. Whole-suite and
multi-file runs were skipped under the shared-host rule, and no heavy-run
lock was bypassed. Offline output checking also passed with pandas 2.3.3/
NumPy 2.3.3 and pandas 3.0.6/NumPy 2.5.4. The [build report](land_residual_run.md)
records commands, independent checks and deliverables.
