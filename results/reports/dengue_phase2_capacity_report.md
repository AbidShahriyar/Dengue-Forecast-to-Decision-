# Capacity-aware dengue resource-planning prototype

## What is now workable

The one-week admission forecast can now be translated into bed-days, average occupied-bed equivalents, IV-fluid litres, and 500-mL-equivalent bags. The operational target remains DGHS reported hospital admissions. A separate 0%, 10%, 25%, and 50% uplift grid represents possible admission-eligible demand not captured in those reports; those uplifts are sensitivity tests, not estimates.

## Selected illustrative planning case

- Forecast: 80% upper one-week admissions forecast = 13,454.9.
- Incomplete-capture sensitivity: +25%.
- Planning admissions = 16,818.6.
- Length of stay = 5 days, producing 84,093 bed-days or 12,013 average occupied-bed equivalents.
- IV planning coefficient = 4.5 L per admission, producing 75,684 L or 151,368 500-mL-equivalent bags.
- Latest-share allocation sums to 16,818.6 planning admissions. The robust three-week divisional envelope sums to 17,809.8, a 5.9% spatial reserve premium because each division receives its maximum recent share.

This is not a clinical prescribing tool or a final purchase order. The clinical coefficients are median-based scenarios from a selected 2019 Group B/C cohort in three Dhaka tertiary hospitals, not national means.

## Sensitivity range

- observed-workload low: 10,649.2 planning admissions; 42,597 bed-days (6,085 average occupied beds); 21,298 L IV fluid (42,597 500-mL equivalents).
- selected planning: 16,818.6 planning admissions; 84,093 bed-days (12,013 average occupied beds); 75,684 L IV fluid (151,368 500-mL equivalents).
- stress high: 20,182.3 planning admissions; 121,094 bed-days (17,299 average occupied beds); 121,094 L IV fluid (242,189 500-mL equivalents).

## Capacity evidence and the admission-censoring issue

The public DGHS 2025 pages yielded 522 facilities and 56,534 approved public beds across four facility classes. 335 of 522 facilities reporting BOR were at or above 100%. This supports treating capacity pressure as plausible, but it does not identify how many dengue patients were denied, diverted, treated elsewhere, or omitted from surveillance.

Accordingly, under-capture is modeled by partial identification: conclusions are checked over a declared uplift set rather than pretending there is a known correction factor. The 2025 approved-bed/BOR table is not subtracted from dengue bed needs because it is annual, all-cause, public-only, and does not report live staffed or free beds.

## What remains blocked

A real constrained allocation optimization still needs facility-week stocks and capacities: usable dengue beds, current IV inventory and replenishment lead times, staff rosters plus activity-time standards, transfer limits, and counts of diversions/denied admissions. Until those arrive, this phase provides defensible requirement scenarios and a robust pre-positioning envelope, not a claim about the exact number of new beds, doctors, or nurses to deploy.

## Proposed research workflow and novelty test

1. Freeze the one-week rolling-origin forecast and its calibrated interval before tuning allocation decisions.
2. Translate every historical forecast origin into bed-day and IV-fluid scenarios using the declared clinical and capture-sensitivity grids.
3. Compare spatial policies retrospectively: latest-week shares, pooled recent shares, and the three-week robust maximum-share envelope.
4. Score decision performance, not only forecast error: realized shortage, unused reserve, worst-division service fraction, and allocation stability.
5. Report conclusions that survive all plausible capture uplifts as partially identified; report conclusions that change with uplift as sensitivity-dependent.
6. Add facility-week stocks, staffed capacity, replenishment, and transfers when obtained, then solve the actual constrained allocation problem and compare it with simple proportional policies.

The publishable contribution is the evaluated chain from calibrated weekly admission uncertainty to equitable, capacity-aware resource decisions under incomplete admission capture. Forecast-to-resource multiplication by itself is not novel enough.

## Reproducibility

Run `python3 work/phase2_capacity/build_phase2_capacity.py` to rebuild from cached official pages. Add `--refresh` to re-download the four public DGHS 2025 detail pages. No anti-bot protection is bypassed; a normal browser user agent is used, and any failed/blocked response stops the refresh rather than being circumvented.
