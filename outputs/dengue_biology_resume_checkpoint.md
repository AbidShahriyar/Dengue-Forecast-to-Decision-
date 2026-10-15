# Biology-informed forecast checkpoint

Last updated: 2026-10-14T23:57:39.042459+06:00

## Status

- Verified weekly target overlay: complete.
- Biology feature construction: complete.
- Rolling-origin 1-, 2-, and 4-week comparison: complete.
- Sequential 52-week, count-scaled conformal recalibration: complete.
- Status: complete.

## Resume

Run `python3 src/build_biology_benchmark.py`. Completed horizon
checkpoints are reused only when the verified target, daily weather, and
configuration hashes match.

## Selected models

 horizon   selected_model  n_common        mae        rmse     wape  mean_bias  peak_week_mae  coverage_80_asymmetric  coverage_80_conformal  coverage_80_conformal_2026  width_80_conformal  interval_score_80_conformal  mae_rank  mae_skill_vs_persistence  wape_skill_vs_persistence
       1 ensemble_log_hgb       124 226.493326  376.217837 0.098342  22.777985     553.297074                0.740385               0.855769                       0.900          830.486440                  1141.492574         1                  0.349820                   0.349820
       2 ensemble_log_hgb       123 389.246950  654.182895 0.168733  45.487042     944.028503                0.725490               0.843137                       0.875         1390.181908                  1939.703153         1                  0.404244                   0.404244
       4 ensemble_log_hgb       121 764.892286 1386.251230 0.332267 304.795676    1494.679109                0.693878               0.867347                       0.875         2423.502732                  3544.537830         1                  0.370921                   0.370921
