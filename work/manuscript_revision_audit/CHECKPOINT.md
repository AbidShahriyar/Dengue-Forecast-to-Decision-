# Manuscript revision feasibility audit

Generated: 2026-10-14T12:59:01.245121+06:00

## Decision

The proposed scientific revisions can be incorporated without collecting new data,
with one important boundary: the paper cannot claim that latent dengue demand or
capacity censoring was estimated.  That portion must remain a declared
partial-identification sensitivity analysis and the title/abstract must say
**potentially capacity-censored**.

After the revision, the work is defensible as a rigorous applied forecasting and
decision-analysis paper.  The current evidence is strongest for a solid Q2-target
submission.  It is not yet a strong Q1 operations claim because the evaluation
covers only three seasons, has no external validation, and lacks facility-week
capacity and unmet-demand outcomes.

## Results that change the manuscript's headline

1. **A simple one-week growth model is the point-estimate winner.** Its WAPE is
   12.0%, compared with 13.1% for baseline HGB.
   The paired MAE advantage is -22.4
   admissions, but its 95% block-bootstrap interval
   [-104.3,
   56.4] crosses zero.  The paper should
   report this as a competitive, parsimonious result rather than a proved
   superiority claim.

2. **Biology-informed constraints remain a credible negative result.** At four
   weeks, constrained biology has MAE 948.9 versus 951.5
   for baseline HGB, but the paired difference is
   30.1 admissions with a 95% interval
   [-65.1,
   143.8].  There is no reliable evidence
   that the biological constraints improve forecast accuracy.

3. **The allocation result is a trade-off, not a general win for adaptive
   reserves.** Point forecast plus pooled shares has mean weekly shortage
   14.0%.  Replacing the forecast total with the
   observed-total oracle reduces this by 5.18 percentage points
   (95% CI 2.74 percentage points to
   7.62 percentage points), about
   36.9%
   of the point-policy mean shortage.

4. **Exact budget matching confirms the reviewer's concern.** The point
   max-share envelope is dominated by its matched fixed reserve: adaptive minus
   fixed mean shortage is 0.25 percentage points.
   Upper-80 pooled and upper-80 envelope reduce unweighted mean shortage by
   0.61 percentage points and
   1.35 percentage points, respectively, but
   worsen admissions-weighted shortage by
   2.05 percentage points and
   1.10 percentage points.  The latter
   differences have positive 95% intervals, so the large-outbreak penalty is not
   merely a point-estimate artifact.

5. **The capacity proxy is not identified as an operational rule.** The
   decision-aligned minimax weight changes across latent-demand and reserve
   assumptions.  The defensible phrasing is “no stable evidence of benefit,” not
   “the proxy failed.”

## Censoring check

Among complete weeks there are 10 adjacent top-decile
high-to-high transitions, of which 3 change by no more than 5%.
This descriptive plateau check cannot distinguish capacity saturation from epidemic
dynamics, reporting practices, or chance.  It therefore does not repair the
identification problem; the softened claim is mandatory.

## Point-by-point incorporation decision

- **0_numbers — incorporable_now:** Recomputed one-week WAPE is 12.0%; point-policy mean shortage is 14.0%; oracle gap is 5.18 percentage points. Action: Replace all superseded abstract, table, figure, and narrative values from one frozen output set.
- **2.1_reserve_frontier — incorporable_now_with_changed_headline:** At equal mean budget, the point max-share envelope is dominated; upper-bound rules lower unweighted mean shortage but worsen demand-weighted shortage. Action: Lead with the fixed-reserve benchmark and explicitly report the mean-versus-surge trade-off.
- **2.2_full_model_comparison — incorporable_now:** Log-growth is best at one week (12.0% versus 13.1% for HGB); its paired MAE difference CI still crosses zero. Action: Show all primary models, skill versus persistence, interval metrics, and place the unstable negative-binomial result in context.
- **2.3_selection_bias — incorporable_now:** A prior-origin-only rolling selector has one-week WAPE 13.2%, versus 12.0% for the globally best backtest model. Action: Call globally chosen models 'best in backtest' and report the rolling-selection sensitivity.
- **2.4_season_and_intensity — incorporable_now_but_limited:** Log-growth beats HGB by MAE in 3/3 one-week seasons; peak-quartile WAPE is 10.2% and trough-quartile WAPE is 33.1%. Action: Add per-season and peak/trough results and retain the three-season/no-external-validation limitation.
- **2.5_budget_ratio — incorporable_now:** Mean weekly budget/actual is 108.2%, but the ratio of aggregate sums is 101.7%; division totals are 4.5% above matched daily totals in aggregate. Action: Explain the mean-of-weekly-ratios effect and the division-versus-daily aggregation mismatch.
- **2.6_capacity_proxy — incorporable_only_as_sensitivity:** Decision-aligned minimax weights vary with assumed latent uplift and reserve; selected nontrivial weights include [0.0, 0.25, 0.4, 0.65, 0.8, 1.0]. Action: Use 'no stable evidence of benefit'; do not claim that the proxy failed or that one weight is identified.
- **2.7_scenario_grid — incorporable_now:** Ten spatial profiles x four latent uplifts x five reserve levels x four policies = 800 profile-level combinations and 80,800 week-policy rows. Action: Add the generated scenario-definition table to the supplement and reconcile Methods/Results counts.
- **2.8_forecast_origin — incorporable_now:** The 4-10 October 2026 origin week contains seven published admission days; the 11-17 October target was forecast after the origin week closed. Action: State the information cutoff explicitly and separate it from the later one-day partial target observation.
- **2.9_capacity_censoring — requires_softened_claim:** Admissions-only series cannot identify denied or displaced demand; a plateau diagnostic is descriptive, not proof of censoring. Action: Use 'potentially capacity-censored' and present latent demand as partial-identification sensitivity, not an estimated outcome.

## What the approved manuscript rewrite should do

- Replace every old abstract/result value from the frozen rerun outputs.
- Reframe the one-week result around the parsimonious growth baseline and retain
  HGB/biology models as comparative tests.
- Present biology-informed forecasting as a prespecified negative result with
  uncertainty, not as the selected operational model.
- Make the fixed-reserve frontier the primary allocation figure; show adaptive
  rules as budget-matched markers and report both mean and demand-weighted shortage.
- Move latent-demand/capacity-proxy findings to a clearly labelled robustness or
  partial-identification section.
- Apply the proposed rhetorical, table, figure, author-summary, and reporting
  improvements only after these numerical conclusions are accepted.

## Data still worth obtaining, but not required for the scoped paper

The requested presentations, admissions, refusals/diversions/waitlists/transfers,
staffed and occupied dengue beds, and admission/discharge timestamps would enable a
new study that estimates unmet demand and physical resource requirements.  They are
not a minor patch to the present admissions-allocation paper; they support a stronger
follow-on causal/operations study.
