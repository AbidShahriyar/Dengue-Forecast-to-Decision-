# Missing-value protocol for weekly dengue forecasting

## Classify before treating

1. **Structural absence:** historical daily dengue values do not exist publicly for long periods. Leave these outcomes missing.
2. **Incomplete target week:** one or more daily target values are absent. Exclude the week unless the source confirms that absence means zero.
3. **Short predictor gap:** impute inside each training window only, add a missingness indicator, and compare against a model that natively handles missing predictors.
4. **Reporting delay:** do not call preliminary recent counts missing-at-random. Estimate a reporting-delay/nowcasting model only from data containing event and report/revision dates.

## Prohibited operations

- Do not convert monthly totals to weekly values.
- Do not linearly interpolate outbreak counts across the 2020-2021 blackout.
- Do not fit an imputer once on the full dataset before cross-validation.
- Do not score a model against imputed outcomes.
- Do not use two-sided smoothing at a historical forecast origin unless it is explicitly labeled retrospective interpolation.

## Allowed sensitivity analyses

- Native missing-value models with explicit missingness indicators.
- Training-fold-only median or seasonal imputation of covariates.
- State-space filtering for latent-state estimation, with uncertainty propagated and observed/imputed status retained.
- Artificial masking experiments on observed predictor blocks to compare imputation methods by gap length.
