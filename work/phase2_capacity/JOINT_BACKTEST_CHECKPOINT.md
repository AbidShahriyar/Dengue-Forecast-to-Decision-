# Joint forecast-allocation checkpoint

Last updated: 2026-10-14 (Asia/Dhaka)

## Completed

- Selected one-week model `ensemble_log_hgb` joined to DGHS division-share policies.
- Confirmed DGHS Sunday-Saturday Wxx mapping with week 1 containing 1 January.
- Primary and isolated-low sensitivity datasets; no missing target was imputed.
- Point, upper-80%, and observed-total diagnostic forecast levels.
- Paired year-stratified moving-block bootstrap and a fixed-reserve frontier.
- Exact aggregate-budget matching with intervals for mean, weighted, and surge shortage.
- HGB, prospective rolling-selector, and direct-divisional forecast sensitivities.

## Primary causal-alignment results

- Point forecast + pooled shares: 14.525% mean shortage; 96.762% total budget/total actual.
- Point forecast + max-share envelope: 10.327% mean shortage; 104.618% total budget/total actual.
- Upper-80% + pooled shares: 5.663% mean shortage; 111.913% total budget/total actual.

## Decision status

- Workable research prototype: yes.
- Week-calendar blocker: resolved using the confirmed Sunday-Saturday definition.
- Routine baseline for further work: point forecast + trailing 3-week pooled shares.
- Reserve policy remains a cost trade-off; no optimum is asserted without shortage, holding, expiry, transfer, and lead-time costs.

## Resume

1. `python3 src/backtest_joint_forecast_allocation.py`
2. `node work/phase2_capacity/build_phase2_workbook.mjs`
3. Next data collection: facility-week usable dengue beds, IV inventory/lead time, staff rosters or WISN inputs, transfers, and denied/diverted admissions.
