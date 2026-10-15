# Phase 3 publication viability and capacity-censoring stress test

Run date: 2026-10-14

## Decision

Proceed as a Q2-caliber research pathway. The completed forecast and allocation
backtests are strong enough for an applied paper, and this phase adds a more
defensible question: how should resources be allocated when hospitalized dengue
surveillance is a capacity-censored lower bound? A Q1 claim is not yet supported.
Recent forecast-to-allocation and stochastic hospital-allocation papers already
cover the generic methodological territory. Q1 would require real facility-level
validation of denied/diverted demand, staffed capacity, patient flow, inventory,
or decision costs.

Quartile placement is not guaranteed and depends on the eventual journal,
reviewers, final methods, and data. The rating here concerns the study design,
not an acceptance prediction.

## Why the new question matters

DGHS counts are hospitalized, test-positive cases in reporting facilities. They
are not all infections or even all people who sought care. Bangladesh literature
also documents incomplete reporting and multiple referrals. Therefore the
reported count is valid as an observed admission workload but may be a lower
bound for admission-eligible demand during congestion.

The model deliberately does not convert the much larger total-infection
under-reporting multipliers into bed demand. Infection under-ascertainment and
capacity-denied admission are different mechanisms.

## Public operational evidence recovered

The TIB field report gives a September 2023 snapshot for ten districts. Six have
numeric allocated dengue-seat counts. Across only those six locations, the table
reports 1,226 patients per day and 3,470 allocated dengue seats.
Using the report's adjacent 4-5-day stay relationship implies 4,904 to
6,130 concurrent patients, or an illustrative shortfall of 1,434
to 2,660 places. This is evidence of severe pressure, not a national
estimate: the source label 'patients per day' is ambiguous, allocated seats are
not verified staffed beds, and the selected districts are not a census.

WHO's first 2023 situation report independently states that COVID-19 beds were
repurposed for dengue and that reported admissions placed additional pressure on
the health system. The official reports also yield a reproducible sentinel panel of 130 hospital-week rows across 20 hospitals and 7 consecutive peak-season weeks (weeks 39-45). Issue 7 publishes only five hospitals per group; the other issues publish ten. All 100 comparable consecutive cumulative-case differences reconcile exactly to the reported weekly cases. The panel is suitable for descriptive triangulation, not independent model training or validation, and contains no staffed-bed, occupancy, refusal, transfer, stock, or staffing fields.

## Stress-test design

For each of 101 one-week hindcast targets, observed divisional admissions are
treated as lower bounds. Latent admission-eligible demand is varied from +0% to
+100%, using uniform, 2025 public-bed-occupancy-linked, inverse, and rotated
spatial patterns. This is a partial-identification sensitivity analysis, not an
imputation. Four fixed-budget allocation rules are compared over reserves from
0% to 100% above the national point forecast.

The capacity-prior policies use 2025 annual public-facility bed occupancy only as
a weak spatial stress proxy. They are never described as measured dengue
capacity, free beds, or historical truth for 2024.

In one prespecified illustration (+50% maximum latent uplift, capacity-linked
pattern, and a budget 50% above the point forecast), pooled reported-demand
shares have 9.95% mean shortage and 58.93% mean
worst-division service. The full capacity-prior rule has 9.78% mean
shortage and 60.73% mean worst-division service. The apparent
shortage gain is only 0.171 percentage points.

That rule fails the misspecification test. Across every spatial stress pattern in
the same block, its average shortage difference from pooled allocation is
+0.003 percentage points and its worst scenario-week difference
is +1.395 percentage points. A minimax mixture audit selected
zero weight on the capacity proxy in 100.0% of target/uplift
combinations. The capacity-prior reweighting is therefore rejected as the
default policy. Until censoring is measured, reserve size can be studied, but
spatial reweighting should remain anchored to observed pooled shares.

## Novelty position

- RAFT-A (Transportation Research Part E, 2026) already joins outbreak forecasts,
  hospitalization demand, predictive distributions, newsvendor quantiles, and
  hospital bed activation. A generic two-phase forecast/allocation contribution
  is no longer new enough.
- Jiang et al. (Transportation Research Part E, 2024) already propose a rolling
  epidemic prediction and collaborative resource-allocation framework.
- Dengue patient-allocation models already optimize assignment among hospitals,
  and a 2026 Bangladesh preprint already proposes a SEIR-informed stochastic,
  multi-objective dengue emergency supply chain.
- The defensible gap is explicit handling of hospitalized surveillance as a
  censored operational signal, with partial identification, fairness, forecast
  uncertainty, and prospective facility validation in a resource-limited dengue
  setting.

## Claims allowed now

1. Weekly national dengue forecasts can be propagated to divisional allocation
   and evaluated out of sample.
2. Resource conclusions are materially sensitive to forecast uncertainty,
   spatial-share instability, and assumptions about unobserved
   admission-eligible demand.
3. Public 2023 evidence demonstrates severe seat pressure in selected districts.
4. The public annual bed-occupancy proxy does not justify changing divisional
   shares: its small assumption-matched gain disappears under spatial
   misspecification, and the ambiguity-minimax choice retains pooled shares.

Do not claim that the model estimates the true number of refused patients, the
true staffed-bed shortfall, an optimal IV order, or the required number of doctors
and nurses.

## Smallest data package that changes the paper materially

The highest-value addition is one or two outbreak seasons from 3-5 sentinel
hospitals with daily: total dengue presentations, admissions, refused/no-bed or
diverted cases, staffed dengue beds, occupied dengue beds, and transfers. Add
admission/discharge timestamps if possible. This would identify the censoring
mechanism and validate occupancy forecasts. It is the single most important
upgrade from a strong Q2 sensitivity paper toward a Q1 operations paper.

For IV-fluid optimization, facility-day opening stock, receipts, use/issues,
stockouts, and order-to-delivery lead times are indispensable. For staffing,
shift rosters or nurse-hours and doctor-hours are indispensable. Neither can be
recovered credibly from admissions alone.

## Reproducibility

Run `python3 work/phase3_latent_demand/build_phase3_latent_demand.py` from the
project root. The analysis is deterministic and records input/output hashes in
its manifest.

## Principal sources

- Transparency International Bangladesh. *Dengue Crisis Prevention and Control*,
  2023, pp. 18-20. https://ti-bangladesh.org/images/2023/report/dengue-crisis/Executive-Study-EN-Dengue-Crisis-Prevention-and-Control-TIB.pdf
- WHO Bangladesh. *Dengue Situation Report, Issue 1*, 28 August 2023.
  https://cdn.who.int/media/docs/default-source/searo/bangladesh/dengue-sitrep/dengue-sitrep-issue-1.pdf
- WHO Bangladesh. *Dengue update 2023*, official situation-report index for
  issues 6-12. https://www.who.int/bangladesh/emergencies/dengue-update-2023
- Soto-Ferrari M, El-Adle AM, Bhattacharyya K. RAFT-A, 2026.
  https://doi.org/10.1016/j.tre.2026.105090
- Jiang J et al. Data-driven collaborative healthcare resource allocation in
  pandemics, 2024. https://doi.org/10.1016/j.tre.2024.103828
- *Solving Patient Allocation Problem during an Epidemic Dengue Fever Outbreak
  by Mathematical Modelling*, 2022.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8775972/
- Sharmin S. *Dengue in Bangladesh: assessment of the influence of climate and
  under-reporting*, 2017. https://openresearch-repository.anu.edu.au/handle/1885/131667
