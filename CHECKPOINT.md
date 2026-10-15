# Reproducibility checkpoint

Release prepared 15 October 2026 (Asia/Dhaka).

## Frozen manuscript results

- One-week ensemble: 124 targets, MAE 226.5, WAPE 9.8%, 80% interval coverage 85.6%.
- Two-week ensemble: MAE 389.2, WAPE 16.9%.
- Four-week ensemble: MAE 764.9, WAPE 33.2%.
- One-week ensemble versus baseline HGB: paired MAE difference −71.8 admissions; 95% moving-block interval −149.5 to −9.2.
- Primary point-plus-pooled allocation: 101 weeks, mean shortage 14.5%, admissions-weighted shortage 11.1%, 307.3 shortfalls per week, and aggregate allocation/observed ratio 96.8%.
- Observed-total oracle gap: 5.67 percentage points; 95% interval 3.08–8.36.
- The 124 forecast targets and 101 allocation weeks are distinct denominators: allocation additionally requires three preceding within-year divisional weeks and excludes the partial 2026 week 41.

## Resume order

Use committed outputs unless an analytical rebuild is necessary:

1. `python3 src/verify_repository.py`
2. `make assets`
3. `make paper` if LaTeX is installed.

For a complete analytical rebuild, run `make full`. Network collection scripts are intentionally separate.

## Data needed for a stronger follow-on study

- total suspected/confirmed presentations;
- admissions and refused/diverted/waitlisted/transferred patients;
- staffed and occupied dengue beds;
- admission/discharge timestamps;
- IV-fluid inventory, receipts, issues, stockouts, and lead times;
- physician/nurse shift hours and activity-time standards.

These fields are not required for the present reported-admissions forecasting and allocation-sensitivity paper. They are required for claims about true unmet demand and exact physical resources.
