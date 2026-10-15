# Compact codebook

## Weekly dengue fields

- `week_year`, `week_number`: DGHS reporting year and week number.
- `week_start`, `week_end`: Sunday and Saturday bounds.
- `admissions`: reported hospitalized dengue admissions for a complete week.
- `target_status`: whether the target is observed, incomplete, or unavailable.
- `target_source`: source used for the weekly total.

## Forecast outputs

- `horizon` or `horizon_weeks`: number of weeks from forecast origin to target.
- `prediction` / `point_forecast_admissions`: point forecast.
- `lower_80_conformal`, `upper_80_conformal`: sequential, locally scaled 80% planning interval with a documented 10% safety expansion.
- `actual`: observed complete weekly admissions used for scoring.
- `model`: candidate forecast model.

## Allocation outputs

- `mean_shortage_fraction`: admissions not covered by an allocation, divided by realized divisional admissions and averaged over target weeks.
- `mean_unused_fraction_of_allocation`: unused allocation divided by allocation.
- `mean_budget_to_actual_ratio`: total allocation divided by observed divisional admissions.
- `mean_worst_division_service_fraction`: mean across weeks of the minimum service fraction among divisions.
- `full_division_coverage_rate`: fraction of weeks in which every division was fully covered.
- `latent_uplift_max`: maximum proportional increase over reported admissions allowed in a censoring scenario.
- `truth_profile`: assumed spatial pattern of the unobserved demand uplift.

All allocation quantities are admission-service-equivalent units unless explicitly converted into bed-days or IV-fluid scenario quantities.

