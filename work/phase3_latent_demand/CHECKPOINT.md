# Phase 3 latent-demand checkpoint

Last updated: 2026-10-14

## Completed

- Transcribed the September 2023 TIB dengue seat/ICU/patient snapshot without
  converting unavailable values to zero.
- Calculated transparent 4-5-day concurrent-patient stress ranges for locations
  with numeric allocated dengue seats.
- Ran a deterministic 101-week partial-identification stress test over four
  latent-demand uplifts, ten spatial patterns, five reserve levels, and four
  allocation rules.
- Rejected the capacity-prior spatial reweighting as a default: a minimax
  mixture audit selected zero proxy weight across every target/uplift pair.
- Produced a publication viability assessment and a prioritized minimum data
  request.
- Recovered and audited the WHO issues 6-12 sentinel-hospital panel: 130 rows,
  20 hospitals, and seven peak-season weeks. It is descriptive triangulation,
  not an adequate facility-model training set.

## Interpretation limit

The latent-demand scenarios are assumptions, not estimates. No public source
found in this phase reports total dengue presentations, admitted patients,
refused/no-bed cases, staffed dengue beds, and occupancy together by facility
and day. Therefore the model supports sensitivity and robustness claims only.

## Resume

1. Collect official WHO reports with the bundled Python runtime:
   `python3 src/collect_who_sitreps.py`
2. Extract and audit the sentinel panel:
   `python3 src/extract_who_sentinel_panel.py`
3. Rebuild the stress test and report:
   `python3 work/phase3_latent_demand/build_phase3_latent_demand.py`
