# Joint forecast-allocation hindcast

Generated: 2026-10-14

## Main result

The joint workflow is technically workable as a research prototype. Using the selected one-week ensemble_log_hgb forecast, confirmed DGHS Sunday-Saturday alignment, and trailing 3-week pooled division shares, the 101-week hindcast had 14.5% mean divisional shortage and 38.8% P90 shortage. Mean absolute shortfall was 307.3 admissions per week, and total allocation was 96.8% of total division-reported admissions.

With the observed national total supplied as a diagnostic oracle, pooled shares alone produced 8.9% mean shortage. The difference to the joint point-forecast result shows that national forecast and aggregation mismatch add material operational error beyond the spatial-share problem.

## Reserve trade-offs

| Strategy | Mean shortage | P90 shortage | Mean unused allocation | Total budget / total actual | Full division coverage |
|---|---:|---:|---:|---:|---:|
| Point forecast + pooled shares | 14.5% | 38.8% | 11.5% | 96.8% | 6.9% |
| Point forecast + max-share envelope | 10.3% | 30.2% | 17.4% | 104.6% | 14.9% |
| Upper 80% bound + pooled shares | 5.7% | 17.5% | 26.8% | 111.9% | 23.8% |
| Upper 80% bound + max-share envelope | 3.3% | 12.3% | 33.0% | 121.3% | 38.6% |

A fixed 20% uplift on the point forecast reduced mean shortage to 8.6% with an aggregate budget of 116.1% of actual. A 30% uplift reduced it to 6.9%, with a mean budget ratio of 125.8%. These are trade-off points, not recommended optima; choosing one requires shortage and holding-cost values.

## Week alignment

DGHS Wxx labels and all national/weather aggregates use Sunday-Saturday weeks; week 1 contains 1 January. The confirmed example 9-15 August 2026 = W33 is reproduced by the calendar function.

## Paired uncertainty comparisons

- Point forecast + pooled shares minus Point forecast + latest share: mean shortage difference -0.63%; moving-block bootstrap 95% interval [-1.29%, -0.02%]. Mean budget-ratio difference -0.0%.
- Point forecast + normalized max shares minus Point forecast + pooled shares: mean shortage difference +0.61%; moving-block bootstrap 95% interval [+0.34%, +0.90%]. Mean budget-ratio difference -0.0%.
- Point forecast + max-share envelope minus Point forecast + pooled shares: mean shortage difference -4.20%; moving-block bootstrap 95% interval [-4.96%, -3.48%]. Mean budget-ratio difference +15.5%.
- Upper 80% bound + pooled shares minus Point forecast + pooled shares: mean shortage difference -8.86%; moving-block bootstrap 95% interval [-10.39%, -7.20%]. Mean budget-ratio difference +47.6%.
- Upper 80% bound + max-share envelope minus Point forecast + max-share envelope: mean shortage difference -7.05%; moving-block bootstrap 95% interval [-8.49%, -5.65%]. Mean budget-ratio difference +57.3%.
- Point forecast + pooled shares minus Observed-total oracle + pooled shares: mean shortage difference +5.67%; moving-block bootstrap 95% interval [+3.08%, +8.36%]. Mean budget-ratio difference +6.2%.

## Interpretation

The point forecast with pooled shares is the lowest-complexity operational baseline. The point-plus-envelope strategy is a moderate contingency option. Using the full upper conformal bound every week substantially reduces shortage, but it also produces large unused allocations and is too conservative to call a routine allocation rule without explicit inventory, expiry, transfer, and shortage costs.

## Limitations

- The analysis uses only one-week forecast targets with non-missing conformal intervals; the early 2024 calibration period is excluded.
- The Sunday-Saturday Wxx convention was supplied by the study team and verified against the example 9-15 August 2026 = W33; an official machine-readable calendar table was not available.
- The forecast target and spatial actuals are both DGHS hospital-admission aggregates, but their totals can differ because they come from different dashboard series and calendar conventions.
- Admissions omit unknown unmet, diverted, non-hospital, and community demand.
- Admission-equivalent allocations do not prove facility feasibility. Usable dengue beds, IV stocks, lead times, workforce, referrals, transport, and transfer constraints remain absent.
