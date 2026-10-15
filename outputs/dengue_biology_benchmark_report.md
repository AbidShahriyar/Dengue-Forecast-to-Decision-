# Biology-informed weekly dengue forecast benchmark

Generated: 2026-10-14T19:54:57.848923+06:00

## Result

- 1-week: **ensemble_log_hgb**, MAE 226.5, WAPE 9.8%, MAE improvement versus baseline 24.7%, planning-interval coverage 85.6% overall and 90.0% in the 2026 evaluation period.
- 2-week: **ensemble_log_hgb**, MAE 389.2, WAPE 16.9%, MAE improvement versus baseline 17.0%, planning-interval coverage 84.3% overall and 87.5% in the 2026 evaluation period.
- 4-week: **ensemble_log_hgb**, MAE 764.9, WAPE 33.2%, MAE improvement versus baseline 19.6%, planning-interval coverage 86.7% overall and 87.5% in the 2026 evaluation period.

The comparison used identical expanding-window forecast origins, target weeks,
and predictor missingness rules. A biology-informed model is not retained merely
because it is more interpretable; it must improve out-of-sample performance.

## Biology encoded

- Nonnegative counts through a Poisson loss.
- A transparent Ae. aegypti thermal-suitability feature that is zero at 17.8 C
  and 34.6 C and peaks at 29.1 C, following the posterior-mean limits reported
  by Mordecai et al. (2017).
- Distributed weather lags from zero to twelve weeks.
- Heat-stress, heavy-rain, humidity, and temperature-rain interaction features.
- In the constrained sensitivity model, forecasts are monotonically increasing
  in thermal suitability and decreasing in heat stress, holding all other
  predictors fixed.

Raw temperature and rainfall are not forced to have globally positive effects.
Suitability is not sufficient for case growth, and heavy rainfall can flush
immature mosquitoes. The triangular curve is explicitly a sensitivity
approximation, not a reproduction of the original mechanistic R0(T) equation.

Scientific references:

- https://doi.org/10.1371/journal.pntd.0005568
- https://doi.org/10.1371/journal.pntd.0007213
- https://doi.org/10.1371/journal.pcbi.1013540

## Recalibrated current research forecasts

- Week beginning 2026-10-11: 12,932 admissions (80% conformal interval 11,836-14,028).
- Week beginning 2026-10-18: 12,964 admissions (80% conformal interval 11,311-14,616).
- Week beginning 2026-11-01: 9,585 admissions (80% conformal interval 6,720-12,450).

Intervals use sequential, locally scaled conformal calibration. At each
historical origin, only forecast errors whose target week had already resolved
were eligible. Absolute errors are divided by the square root of the point
forecast (a count-data variance stabilizer), the latest 52 eligible errors are
used, and the restored interval radius receives a transparent 10% planning
safety expansion. The nominal target is 80%; time-series dependence means this
is an empirical planning interval, not an exchangeability-based coverage
guarantee.

## Interpretation limit

This is biology-informed feature engineering with a constrained sensitivity
model, not a fully identified human-mosquito SEI-SEIR model. A defensible full
hybrid requires serotype/immunity, mosquito abundance, intervention, and
infection-to-admission observation data that are not currently available.
