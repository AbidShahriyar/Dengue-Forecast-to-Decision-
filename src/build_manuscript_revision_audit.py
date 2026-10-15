#!/usr/bin/env python3
"""Build an analysis-only audit for the proposed manuscript revisions.

This script reads completed forecast and allocation outputs.  It does not edit
the manuscript or regenerate figures.  Its purpose is to determine which
reviewer requests are supported by the existing data before prose is revised.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
WORK = ROOT / "work/manuscript_revision_audit"

PREDICTIONS = OUTPUTS / "dengue_biology_backtest_predictions.csv"
METRICS = OUTPUTS / "dengue_biology_benchmark_metrics.csv"
SEASONAL = OUTPUTS / "dengue_biology_benchmark_metrics_by_season.csv"
FORECAST_BOOTSTRAP = OUTPUTS / "dengue_biology_paired_bootstrap.csv"
ALLOCATION_SUMMARY = OUTPUTS / "dengue_phase2_joint_backtest_summary.csv"
ALLOCATION_DETAIL = OUTPUTS / "dengue_phase2_joint_backtest_detail.csv"
ALLOCATION_BOOTSTRAP = OUTPUTS / "dengue_phase2_joint_bootstrap_comparisons.csv"
BUDGET_MATCHED = OUTPUTS / "dengue_phase2_joint_budget_matched_comparisons.csv"
RESERVE_FRONTIER = OUTPUTS / "dengue_phase2_joint_reserve_frontier.csv"
DECISION_MINIMAX = OUTPUTS / "dengue_phase3_decision_minimax_summary.csv"
NATIONAL_WEEKLY = ROOT / "work/weekly_covariates/dengue_weekly_national.csv"

PRIMARY_MODELS = [
    "last",
    "mean4",
    "seasonal52",
    "log_growth4",
    "negative_binomial_glm",
    "baseline_hgb",
    "bio_hgb",
    "bio_constrained",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def pp(value: float, digits: int = 2) -> str:
    return f"{100.0 * value:.{digits}f} percentage points"


def forecast_by_intensity(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare models on common peak, middle, and trough target weeks."""
    rows: list[dict] = []
    for horizon, horizon_data in predictions.groupby("horizon", sort=True):
        pivot = horizon_data.pivot_table(
            index="target_week_start", columns="model", values="prediction"
        )
        common_targets = pivot.dropna(subset=PRIMARY_MODELS).index
        common = horizon_data[
            horizon_data["target_week_start"].isin(common_targets)
            & horizon_data["model"].isin(PRIMARY_MODELS)
        ].copy()
        actual = common.drop_duplicates("target_week_start").set_index(
            "target_week_start"
        )["actual"]
        low = float(actual.quantile(0.25))
        high = float(actual.quantile(0.75))
        labels = pd.Series("middle_50pct", index=actual.index)
        labels.loc[actual.le(low)] = "trough_bottom_quartile"
        labels.loc[actual.ge(high)] = "peak_top_quartile"
        common["intensity_stratum"] = common["target_week_start"].map(labels)
        for (stratum, model), group in common.groupby(
            ["intensity_stratum", "model"], sort=True
        ):
            error = group["prediction"] - group["actual"]
            interval = group.dropna(
                subset=["lower_80_conformal", "upper_80_conformal"]
            ).copy()
            if len(interval):
                interval_score = (
                    interval["upper_80_conformal"] - interval["lower_80_conformal"]
                    + 10.0
                    * (interval["lower_80_conformal"] - interval["actual"])
                    * interval["actual"].lt(interval["lower_80_conformal"])
                    + 10.0
                    * (interval["actual"] - interval["upper_80_conformal"])
                    * interval["actual"].gt(interval["upper_80_conformal"])
                )
                absolute = (interval["prediction"] - interval["actual"]).abs()
                # WIS with the median and one central 80% interval (alpha=0.20).
                wis = (0.5 * absolute + 0.10 * interval_score) / 1.5
            else:
                interval_score = pd.Series(dtype=float)
                wis = pd.Series(dtype=float)
            rows.append({
                "horizon": int(horizon),
                "intensity_stratum": stratum,
                "trough_threshold_admissions": low,
                "peak_threshold_admissions": high,
                "model": model,
                "target_weeks": int(len(group)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "wape": float(error.abs().sum() / group["actual"].sum()),
                "mean_bias": float(error.mean()),
                "interval_target_weeks": int(len(interval)),
                "coverage_80_conformal": float(
                    interval["covered_80_conformal"].mean()
                ) if len(interval) else np.nan,
                "mean_width_80_conformal": float(
                    (interval["upper_80_conformal"] - interval["lower_80_conformal"]).mean()
                ) if len(interval) else np.nan,
                "mean_interval_score_80": float(interval_score.mean())
                if len(interval) else np.nan,
                "mean_weighted_interval_score_80": float(wis.mean())
                if len(interval) else np.nan,
            })
    return pd.DataFrame(rows).sort_values(
        ["horizon", "intensity_stratum", "mae"]
    ).reset_index(drop=True)


def scenario_grid() -> pd.DataFrame:
    definitions = {
        "uniform": "Equal proportional latent-demand uplift in all eight divisions.",
        "capacity_linked": "Uplift score increases with the normalized annual occupancy proxy.",
        "inverse_capacity": "Uplift score decreases with the normalized annual occupancy proxy.",
        **{
            f"rotation_{shift}": (
                f"Capacity-linked score rotated by {shift} division positions to stress-test "
                "spatial misspecification."
            )
            for shift in range(1, 8)
        },
    }
    rows = []
    for profile, definition in definitions.items():
        rows.append({
            "truth_profile": profile,
            "definition": definition,
            "latent_uplift_max_grid": "0%, 25%, 50%, 100%",
            "reserve_uplift_grid": "0%, 25%, 50%, 75%, 100%",
            "allocation_policies": (
                "pooled reported; normalized maximum share; "
                "half-weight capacity prior; full-weight capacity prior"
            ),
            "profile_level_combinations": 4 * 5 * 4,
            "all_profile_level_combinations": 10 * 4 * 5 * 4,
            "historical_target_weeks": 101,
            "week_policy_scenario_rows": 10 * 4 * 5 * 4 * 101,
        })
    return pd.DataFrame(rows)


def censoring_diagnostic(national: pd.DataFrame) -> pd.DataFrame:
    """A descriptive plateau check that cannot establish capacity censoring."""
    frame = national[national["admission_days_published"].eq(7)].copy()
    frame["week_start"] = pd.to_datetime(frame["week_start"])
    frame = frame.sort_values("week_start").reset_index(drop=True)
    high_threshold = float(frame["admissions"].quantile(0.90))
    frame["high_week"] = frame["admissions"].ge(high_threshold)
    frame["previous_admissions"] = frame["admissions"].shift()
    frame["previous_high"] = frame["high_week"].shift(fill_value=False)
    frame["contiguous"] = frame["week_start"].diff().dt.days.eq(7)
    high_transitions = frame[
        frame["high_week"] & frame["previous_high"] & frame["contiguous"]
    ].copy()
    high_transitions["relative_change"] = (
        (high_transitions["admissions"] - high_transitions["previous_admissions"]).abs()
        / high_transitions["previous_admissions"].clip(lower=1)
    )
    exact_ties = int(
        high_transitions["admissions"].eq(high_transitions["previous_admissions"]).sum()
    )
    near_plateaus = int(high_transitions["relative_change"].le(0.05).sum())
    year_peaks = frame.groupby("selected_year")["admissions"].max()
    rows = [
        {
            "diagnostic": "complete_weeks",
            "value": float(len(frame)),
            "interpretation": "Weeks with all seven daily admission dates published.",
        },
        {
            "diagnostic": "top_decile_threshold_admissions",
            "value": high_threshold,
            "interpretation": "Descriptive threshold only; not a capacity estimate.",
        },
        {
            "diagnostic": "adjacent_top_decile_transitions",
            "value": float(len(high_transitions)),
            "interpretation": "Adjacent high-to-high transitions available for a plateau check.",
        },
        {
            "diagnostic": "exactly_equal_adjacent_high_transitions",
            "value": float(exact_ties),
            "interpretation": "Exact ties could indicate rounding or saturation but are not causal evidence.",
        },
        {
            "diagnostic": "adjacent_high_transitions_within_5pct",
            "value": float(near_plateaus),
            "interpretation": "Near-flat high transitions; still non-identifying without capacity or denied-admission data.",
        },
        {
            "diagnostic": "distinct_complete_week_admission_values",
            "value": float(frame["admissions"].nunique()),
            "interpretation": "A mechanical hard cap would tend to reduce distinct values near the ceiling.",
        },
        {
            "diagnostic": "annual_peak_coefficient_of_variation",
            "value": float(year_peaks.std(ddof=1) / year_peaks.mean()),
            "interpretation": "Variation in annual peaks is descriptive and does not identify unmet demand.",
        },
    ]
    return pd.DataFrame(rows)


def find_row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    keep = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        keep &= frame[column].eq(value)
    match = frame[keep]
    if len(match) != 1:
        raise RuntimeError(f"Expected one row for {conditions}; found {len(match)}")
    return match.iloc[0]


def build_decisions(
    metrics: pd.DataFrame,
    seasonal: pd.DataFrame,
    forecast_bootstrap: pd.DataFrame,
    allocation_summary: pd.DataFrame,
    allocation_detail: pd.DataFrame,
    allocation_bootstrap: pd.DataFrame,
    matched: pd.DataFrame,
    minimax: pd.DataFrame,
    intensity: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    h1_growth = find_row(metrics, horizon=1, model="log_growth4")
    h1_hgb = find_row(metrics, horizon=1, model="baseline_hgb")
    h1_rolling = find_row(metrics, horizon=1, model="rolling_selected_all")
    h4_bio = find_row(metrics, horizon=4, model="bio_constrained")
    h4_hgb = find_row(metrics, horizon=4, model="baseline_hgb")
    bio_boot = find_row(
        forecast_bootstrap, horizon=4, challenger_model="bio_constrained"
    )
    growth_boot = find_row(
        forecast_bootstrap, horizon=1, challenger_model="log_growth4"
    )
    base = allocation_summary[
        allocation_summary["analysis_rule"].eq("published_primary")
        & allocation_summary["alignment_scenario"].eq("confirmed_dghs")
    ]
    point = find_row(base, forecast_level="point", policy="pooled3")
    oracle = find_row(base, forecast_level="observed_total_oracle", policy="pooled3")
    oracle_gap = find_row(
        allocation_bootstrap,
        candidate_strategy="point__pooled3",
        reference_strategy="observed_total_oracle__pooled3",
    )
    point_detail = allocation_detail[
        allocation_detail["analysis_rule"].eq("published_primary")
        & allocation_detail["alignment_scenario"].eq("confirmed_dghs")
        & allocation_detail["forecast_level"].eq("point")
        & allocation_detail["policy"].eq("pooled3")
    ]
    aggregate_budget_ratio = (
        point_detail["allocation_budget_admission_equivalents"].sum()
        / point_detail["division_actual_admissions"].sum()
    )
    division_daily_ratio = (
        point_detail["division_actual_admissions"].sum()
        / point_detail["mapped_daily_actual_admissions"].sum()
    )
    upper_pooled = find_row(matched, adaptive_rule="upper80_pooled")
    upper_envelope = find_row(matched, adaptive_rule="upper80_max3_envelope")
    point_envelope = find_row(matched, adaptive_rule="point_max3_envelope")
    one_week_seasons = seasonal[
        seasonal["horizon"].eq(1)
        & seasonal["model"].isin(["log_growth4", "baseline_hgb"])
    ].pivot(index="target_year", columns="model", values="mae")
    growth_season_wins = int(
        (one_week_seasons["log_growth4"] < one_week_seasons["baseline_hgb"]).sum()
    )
    peak = find_row(
        intensity,
        horizon=1,
        intensity_stratum="peak_top_quartile",
        model="log_growth4",
    )
    trough = find_row(
        intensity,
        horizon=1,
        intensity_stratum="trough_bottom_quartile",
        model="log_growth4",
    )
    minimax_nonzero = minimax[
        minimax["criterion"].eq("worst_case_shortage")
        & minimax["latent_uplift_max"].gt(0)
        & minimax["budget_uplift_over_point_forecast"].gt(0)
    ]
    distinct_weights = sorted(
        minimax_nonzero["selected_capacity_proxy_weight"].unique().tolist()
    )

    context = {
        "h1_growth": h1_growth,
        "h1_hgb": h1_hgb,
        "h1_rolling": h1_rolling,
        "h4_bio": h4_bio,
        "h4_hgb": h4_hgb,
        "bio_boot": bio_boot,
        "growth_boot": growth_boot,
        "point": point,
        "oracle": oracle,
        "oracle_gap": oracle_gap,
        "aggregate_budget_ratio": aggregate_budget_ratio,
        "division_daily_ratio": division_daily_ratio,
        "upper_pooled": upper_pooled,
        "upper_envelope": upper_envelope,
        "point_envelope": point_envelope,
        "growth_season_wins": growth_season_wins,
        "peak": peak,
        "trough": trough,
        "distinct_weights": distinct_weights,
    }

    rows = [
        {
            "review_item": "0_numbers",
            "status": "incorporable_now",
            "key_result": (
                f"Recomputed one-week WAPE is {pct(h1_growth.wape)}; point-policy mean "
                f"shortage is {pct(point.mean_shortage_fraction)}; oracle gap is "
                f"{pp(oracle_gap.observed_mean_shortage_difference)}."
            ),
            "required_manuscript_action": "Replace all superseded abstract, table, figure, and narrative values from one frozen output set.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.1_reserve_frontier",
            "status": "incorporable_now_with_changed_headline",
            "key_result": (
                "At equal mean budget, the point max-share envelope is dominated; upper-bound "
                "rules lower unweighted mean shortage but worsen demand-weighted shortage."
            ),
            "required_manuscript_action": "Lead with the fixed-reserve benchmark and explicitly report the mean-versus-surge trade-off.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.2_full_model_comparison",
            "status": "incorporable_now",
            "key_result": (
                f"Log-growth is best at one week ({pct(h1_growth.wape)} versus "
                f"{pct(h1_hgb.wape)} for HGB); its paired MAE difference CI still crosses zero."
            ),
            "required_manuscript_action": "Show all primary models, skill versus persistence, interval metrics, and place the unstable negative-binomial result in context.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.3_selection_bias",
            "status": "incorporable_now",
            "key_result": (
                f"A prior-origin-only rolling selector has one-week WAPE {pct(h1_rolling.wape)}, "
                f"versus {pct(h1_growth.wape)} for the globally best backtest model."
            ),
            "required_manuscript_action": "Call globally chosen models 'best in backtest' and report the rolling-selection sensitivity.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.4_season_and_intensity",
            "status": "incorporable_now_but_limited",
            "key_result": (
                f"Log-growth beats HGB by MAE in {growth_season_wins}/3 one-week seasons; "
                f"peak-quartile WAPE is {pct(peak.wape)} and trough-quartile WAPE is {pct(trough.wape)}."
            ),
            "required_manuscript_action": "Add per-season and peak/trough results and retain the three-season/no-external-validation limitation.",
            "blocking_data": "An independent later season or external site is needed for a stronger generalization claim, but not for the present paper.",
        },
        {
            "review_item": "2.5_budget_ratio",
            "status": "incorporable_now",
            "key_result": (
                f"Mean weekly budget/actual is {pct(point.mean_budget_to_actual_ratio)}, "
                f"but the ratio of aggregate sums is {pct(aggregate_budget_ratio)}; division totals "
                f"are {pct(division_daily_ratio - 1.0)} above matched daily totals in aggregate."
            ),
            "required_manuscript_action": "Explain the mean-of-weekly-ratios effect and the division-versus-daily aggregation mismatch.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.6_capacity_proxy",
            "status": "incorporable_only_as_sensitivity",
            "key_result": (
                "Decision-aligned minimax weights vary with assumed latent uplift and reserve; "
                f"selected nontrivial weights include {distinct_weights}."
            ),
            "required_manuscript_action": "Use 'no stable evidence of benefit'; do not claim that the proxy failed or that one weight is identified.",
            "blocking_data": "Facility-week staffed beds, occupancy, refusals, and transfers are required to validate a capacity allocation rule.",
        },
        {
            "review_item": "2.7_scenario_grid",
            "status": "incorporable_now",
            "key_result": "Ten spatial profiles x four latent uplifts x five reserve levels x four policies = 800 profile-level combinations and 80,800 week-policy rows.",
            "required_manuscript_action": "Add the generated scenario-definition table to the supplement and reconcile Methods/Results counts.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.8_forecast_origin",
            "status": "incorporable_now",
            "key_result": "The 4-10 October 2026 origin week contains seven published admission days; the 11-17 October target was forecast after the origin week closed.",
            "required_manuscript_action": "State the information cutoff explicitly and separate it from the later one-day partial target observation.",
            "blocking_data": "none",
        },
        {
            "review_item": "2.9_capacity_censoring",
            "status": "requires_softened_claim",
            "key_result": "Admissions-only series cannot identify denied or displaced demand; a plateau diagnostic is descriptive, not proof of censoring.",
            "required_manuscript_action": "Use 'potentially capacity-censored' and present latent demand as partial-identification sensitivity, not an estimated outcome.",
            "blocking_data": "Presentations, admission decisions, refusals/diversions/waitlists/transfers, staffed beds, occupancy, and timestamps.",
        },
    ]
    return pd.DataFrame(rows), context


def build_report(
    decisions: pd.DataFrame,
    context: dict[str, object],
    censoring: pd.DataFrame,
) -> str:
    h1_growth = context["h1_growth"]
    h1_hgb = context["h1_hgb"]
    h4_bio = context["h4_bio"]
    h4_hgb = context["h4_hgb"]
    bio_boot = context["bio_boot"]
    growth_boot = context["growth_boot"]
    point = context["point"]
    oracle_gap = context["oracle_gap"]
    upper_pooled = context["upper_pooled"]
    upper_envelope = context["upper_envelope"]
    point_envelope = context["point_envelope"]
    plateau = find_row(censoring, diagnostic="adjacent_high_transitions_within_5pct")
    high_transitions = find_row(censoring, diagnostic="adjacent_top_decile_transitions")

    decision_lines = "\n".join(
        f"- **{row.review_item} — {row.status}:** {row.key_result} "
        f"Action: {row.required_manuscript_action}"
        for row in decisions.itertuples()
    )
    return f"""# Manuscript revision feasibility audit

Generated: {datetime.now().astimezone().isoformat()}

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
   {pct(h1_growth.wape)}, compared with {pct(h1_hgb.wape)} for baseline HGB.
   The paired MAE advantage is {growth_boot.challenger_minus_reference_mae:,.1f}
   admissions, but its 95% block-bootstrap interval
   [{growth_boot.mae_difference_95pct_ci_lower:,.1f},
   {growth_boot.mae_difference_95pct_ci_upper:,.1f}] crosses zero.  The paper should
   report this as a competitive, parsimonious result rather than a proved
   superiority claim.

2. **Biology-informed constraints remain a credible negative result.** At four
   weeks, constrained biology has MAE {h4_bio.mae:,.1f} versus {h4_hgb.mae:,.1f}
   for baseline HGB, but the paired difference is
   {bio_boot.challenger_minus_reference_mae:,.1f} admissions with a 95% interval
   [{bio_boot.mae_difference_95pct_ci_lower:,.1f},
   {bio_boot.mae_difference_95pct_ci_upper:,.1f}].  There is no reliable evidence
   that the biological constraints improve forecast accuracy.

3. **The allocation result is a trade-off, not a general win for adaptive
   reserves.** Point forecast plus pooled shares has mean weekly shortage
   {pct(point.mean_shortage_fraction)}.  Replacing the forecast total with the
   observed-total oracle reduces this by {pp(oracle_gap.observed_mean_shortage_difference)}
   (95% CI {pp(oracle_gap.shortage_difference_95pct_ci_lower)} to
   {pp(oracle_gap.shortage_difference_95pct_ci_upper)}), about
   {100 * oracle_gap.observed_mean_shortage_difference / point.mean_shortage_fraction:.1f}%
   of the point-policy mean shortage.

4. **Exact budget matching confirms the reviewer's concern.** The point
   max-share envelope is dominated by its matched fixed reserve: adaptive minus
   fixed mean shortage is {pp(point_envelope.adaptive_minus_fixed_mean_shortage)}.
   Upper-80 pooled and upper-80 envelope reduce unweighted mean shortage by
   {pp(-upper_pooled.adaptive_minus_fixed_mean_shortage)} and
   {pp(-upper_envelope.adaptive_minus_fixed_mean_shortage)}, respectively, but
   worsen admissions-weighted shortage by
   {pp(upper_pooled.adaptive_minus_fixed_demand_weighted_shortage)} and
   {pp(upper_envelope.adaptive_minus_fixed_demand_weighted_shortage)}.  The latter
   differences have positive 95% intervals, so the large-outbreak penalty is not
   merely a point-estimate artifact.

5. **The capacity proxy is not identified as an operational rule.** The
   decision-aligned minimax weight changes across latent-demand and reserve
   assumptions.  The defensible phrasing is “no stable evidence of benefit,” not
   “the proxy failed.”

## Censoring check

Among complete weeks there are {int(high_transitions.value)} adjacent top-decile
high-to-high transitions, of which {int(plateau.value)} change by no more than 5%.
This descriptive plateau check cannot distinguish capacity saturation from epidemic
dynamics, reporting practices, or chance.  It therefore does not repair the
identification problem; the softened claim is mandatory.

## Point-by-point incorporation decision

{decision_lines}

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
"""


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(PREDICTIONS)
    metrics = pd.read_csv(METRICS)
    seasonal = pd.read_csv(SEASONAL)
    forecast_bootstrap = pd.read_csv(FORECAST_BOOTSTRAP)
    allocation_summary = pd.read_csv(ALLOCATION_SUMMARY)
    allocation_detail = pd.read_csv(ALLOCATION_DETAIL)
    allocation_bootstrap = pd.read_csv(ALLOCATION_BOOTSTRAP)
    matched = pd.read_csv(BUDGET_MATCHED)
    minimax = pd.read_csv(DECISION_MINIMAX)
    national = pd.read_csv(NATIONAL_WEEKLY)

    intensity = forecast_by_intensity(predictions)
    scenarios = scenario_grid()
    censoring = censoring_diagnostic(national)
    decisions, context = build_decisions(
        metrics,
        seasonal,
        forecast_bootstrap,
        allocation_summary,
        allocation_detail,
        allocation_bootstrap,
        matched,
        minimax,
        intensity,
    )
    report = build_report(decisions, context, censoring)

    output_files = {
        "forecast_by_intensity": OUTPUTS / "dengue_manuscript_revision_forecast_by_intensity.csv",
        "scenario_grid": OUTPUTS / "dengue_manuscript_revision_scenario_grid.csv",
        "censoring_diagnostic": OUTPUTS / "dengue_manuscript_revision_censoring_diagnostic.csv",
        "decisions": OUTPUTS / "dengue_manuscript_revision_decisions.csv",
        "report": OUTPUTS / "dengue_manuscript_revision_feasibility_report.md",
    }
    intensity.to_csv(output_files["forecast_by_intensity"], index=False)
    scenarios.to_csv(output_files["scenario_grid"], index=False)
    censoring.to_csv(output_files["censoring_diagnostic"], index=False)
    decisions.to_csv(output_files["decisions"], index=False)
    output_files["report"].write_text(report, encoding="utf-8")
    (WORK / "CHECKPOINT.md").write_text(report, encoding="utf-8")

    inputs = [
        PREDICTIONS,
        METRICS,
        SEASONAL,
        FORECAST_BOOTSTRAP,
        ALLOCATION_SUMMARY,
        ALLOCATION_DETAIL,
        ALLOCATION_BOOTSTRAP,
        BUDGET_MATCHED,
        RESERVE_FRONTIER,
        DECISION_MINIMAX,
        NATIONAL_WEEKLY,
    ]
    manifest = {
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(),
        "purpose": "analysis-only manuscript revision feasibility audit",
        "manuscript_edited": False,
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in inputs},
        "outputs": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for name, path in output_files.items()
        },
    }
    (WORK / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
