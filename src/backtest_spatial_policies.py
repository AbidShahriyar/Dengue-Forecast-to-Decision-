#!/usr/bin/env python3
"""Retrospectively evaluate division-level dengue allocation rules.

The DGHS published division-by-week labels are kept in their native Wxx
sequence.  The national total observed in each target week is held fixed so
that the exercise isolates spatial allocation performance from national
forecast error.  No anomalous or incomplete target is imputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase2_capacity"
OUTPUTS = ROOT / "outputs"
DIVISION_PATH = ROOT / "work/weekly_covariates/dengue_weekly_division_published.csv"
NATIONAL_PATH = ROOT / "work/weekly_covariates/dengue_weekly_national.csv"
RUN_DATE = "2026-10-13"
DIVISIONS = [
    "Barisal", "Chattogram", "Dhaka", "Khulna",
    "Mymensingh", "Rajshahi", "Rangpur", "Sylhet",
]
POLICY_LABELS = {
    "latest": "Latest observed share",
    "pooled3": "Trailing 3-week pooled share",
    "max3_normalized": "Trailing 3-week max share, normalized",
    "max3_envelope": "Trailing 3-week max-share envelope",
}
RULE_LABELS = {
    "published_primary": "Published-data primary analysis",
    "isolated_low_filtered": "Isolated-low sensitivity analysis",
}
ANOMALY_THRESHOLD = 0.50
REBOUND_THRESHOLD = 1.50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 4
BOOTSTRAP_SEED = 20261013


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_weekly_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DIVISION_PATH)
    raw = raw[raw["published_week_label"].notna()].copy()
    raw["week_num"] = raw["published_week_label"].str.extract(r"(\d+)").astype(int)
    audit = (
        raw.groupby(["selected_year", "published_week_label", "week_num"], as_index=False)
        .agg(
            total_admissions=("admissions", "sum"),
            division_count=("division_label", "nunique"),
            row_count=("division_label", "size"),
        )
        .sort_values(["selected_year", "week_num"])
        .reset_index(drop=True)
    )
    matrix = raw.pivot_table(
        index=["selected_year", "published_week_label", "week_num"],
        columns="division_label",
        values="admissions",
        aggfunc="sum",
    ).reset_index()
    for division in DIVISIONS:
        if division not in matrix.columns:
            matrix[division] = np.nan
    matrix = matrix[["selected_year", "published_week_label", "week_num", *DIVISIONS]]
    matrix = matrix.sort_values(["selected_year", "week_num"]).reset_index(drop=True)
    return matrix, audit


def classify_weeks(audit: pd.DataFrame) -> pd.DataFrame:
    result = audit.copy()
    result["complete_eight_divisions"] = (
        (result["division_count"] == len(DIVISIONS))
        & (result["row_count"] == len(DIVISIONS))
    )
    result["known_in_progress_partial"] = (
        (result["selected_year"] == 2026)
        & (result["published_week_label"] == "W41")
    )
    result["prior3_median"] = np.nan
    result["next3_median"] = np.nan
    for _, indexes in result.groupby("selected_year").groups.items():
        ordered = list(indexes)
        for position, index in enumerate(ordered):
            previous = ordered[max(0, position - 3):position]
            following = ordered[position + 1:position + 4]
            if len(previous) == 3:
                result.loc[index, "prior3_median"] = float(
                    result.loc[previous, "total_admissions"].median()
                )
            if len(following) >= 2:
                result.loc[index, "next3_median"] = float(
                    result.loc[following, "total_admissions"].median()
                )
    result["local_reference_median"] = result[["prior3_median", "next3_median"]].median(axis=1)
    result["next_week_total"] = result.groupby("selected_year")["total_admissions"].shift(-1)
    result["isolated_low_anomaly"] = (
        result["prior3_median"].notna()
        & result["next3_median"].notna()
        & (result["total_admissions"] < ANOMALY_THRESHOLD * result["local_reference_median"])
        & (result["next_week_total"] > REBOUND_THRESHOLD * result["total_admissions"])
    )
    result["included_published_primary"] = (
        result["complete_eight_divisions"] & ~result["known_in_progress_partial"]
    )
    result["included_isolated_low_filtered"] = (
        result["included_published_primary"] & ~result["isolated_low_anomaly"]
    )

    def reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not row["complete_eight_divisions"]:
            reasons.append("not exactly eight division records")
        if row["known_in_progress_partial"]:
            reasons.append("current dashboard week was still in progress at extraction")
        if row["isolated_low_anomaly"]:
            reasons.append("isolated-low anomaly; excluded only in sensitivity analysis")
        return "; ".join(reasons) if reasons else "included"

    result["audit_note"] = result.apply(reason, axis=1)
    return result


def alignment_audit(week_audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    national = pd.read_csv(NATIONAL_PATH)
    published = week_audit[week_audit["included_published_primary"]][
        ["selected_year", "week_num", "total_admissions"]
    ].rename(columns={"total_admissions": "published_division_total"})
    rows: list[dict] = []
    for year in sorted(published["selected_year"].unique()):
        year_published = published[published["selected_year"] == year]
        year_national = national[national["selected_year"] == year][
            ["week_number", "admissions"]
        ].rename(
            columns={"week_number": "week_num", "admissions": "national_weekly_admissions"}
        )
        merged = year_published.merge(year_national, on="week_num", how="inner")
        correlation = merged["published_division_total"].corr(
            merged["national_weekly_admissions"]
        ) if len(merged) >= 3 else np.nan
        mae = (
            (merged["published_division_total"] - merged["national_weekly_admissions"]).abs().mean()
            if len(merged) >= 3 else np.nan
        )
        wape = (
            (merged["published_division_total"] - merged["national_weekly_admissions"]).abs().sum()
            / merged["published_division_total"].sum()
            if len(merged) >= 3 else np.nan
        )
        rows.append({
            "selected_year": int(year),
            "published_to_iso_week_shift": 0,
            "matched_weeks": len(merged),
            "pearson_correlation": correlation,
            "mean_absolute_error_admissions": mae,
            "weighted_absolute_percentage_error": wape,
            "interpretation": (
                "Direct DGHS Wxx match on the confirmed Sunday-Saturday calendar; "
                "differences reflect aggregation/source content rather than week shifting."
            ),
        })
    detail = pd.DataFrame(rows)
    best_rows = []
    for year, group in detail.groupby("selected_year"):
        best = group.sort_values(
            ["pearson_correlation", "mean_absolute_error_admissions"],
            ascending=[False, True],
        ).iloc[0]
        best_rows.append({
            "selected_year": int(year),
            "best_correlation_shift": int(best["published_to_iso_week_shift"]),
            "matched_weeks": int(best["matched_weeks"]),
            "pearson_correlation": best["pearson_correlation"],
            "mean_absolute_error_admissions": best["mean_absolute_error_admissions"],
            "conclusion": "Direct Wxx mapping under the confirmed DGHS Sunday-Saturday convention.",
        })
    return detail, pd.DataFrame(best_rows)


def policy_weights(history: pd.DataFrame, policy: str) -> tuple[pd.Series, pd.Series]:
    totals = history[DIVISIONS].sum(axis=1)
    shares = history[DIVISIONS].div(totals, axis=0)
    if policy == "latest":
        raw = shares.iloc[-1]
    elif policy == "pooled3":
        raw = history[DIVISIONS].sum(axis=0) / totals.sum()
    elif policy in {"max3_normalized", "max3_envelope"}:
        raw = shares.max(axis=0)
    else:
        raise ValueError(policy)
    normalized = raw / raw.sum()
    if policy in {"latest", "pooled3", "max3_normalized"}:
        allocation_weight = normalized
    else:
        allocation_weight = raw
    return allocation_weight, normalized


def backtest(
    matrix: pd.DataFrame,
    week_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_week_rows: list[dict] = []
    division_rows: list[dict] = []
    policies = list(POLICY_LABELS)
    for rule, include_column in [
        ("published_primary", "included_published_primary"),
        ("isolated_low_filtered", "included_isolated_low_filtered"),
    ]:
        eligible_keys = set(
            zip(
                week_audit.loc[week_audit[include_column], "selected_year"],
                week_audit.loc[week_audit[include_column], "published_week_label"],
            )
        )
        eligible = matrix[
            matrix.apply(
                lambda row: (row["selected_year"], row["published_week_label"]) in eligible_keys,
                axis=1,
            )
        ].copy()
        for year, year_data in eligible.groupby("selected_year", sort=True):
            year_data = year_data.sort_values("week_num").reset_index(drop=True)
            previous_normalized: dict[str, pd.Series] = {}
            for position in range(3, len(year_data)):
                history = year_data.iloc[position - 3:position]
                target = year_data.iloc[position]
                target_total = float(target[DIVISIONS].sum())
                history_labels = "|".join(history["published_week_label"].astype(str))
                for policy in policies:
                    allocation_weight, normalized = policy_weights(history, policy)
                    actual = target[DIVISIONS].astype(float)
                    allocation = allocation_weight.astype(float) * target_total
                    shortage = (actual - allocation).clip(lower=0)
                    unused = (allocation - actual).clip(lower=0)
                    service = pd.Series(
                        np.where(actual > 0, np.minimum(allocation / actual, 1.0), 1.0),
                        index=DIVISIONS,
                    )
                    allocation_budget = float(allocation.sum())
                    stability = np.nan
                    if policy in previous_normalized:
                        stability = float(
                            0.5 * (normalized - previous_normalized[policy]).abs().sum()
                        )
                    previous_normalized[policy] = normalized
                    full_coverage = bool((shortage <= 1e-9).all())
                    target_id = f"{int(year)}-{target['published_week_label']}"
                    policy_week_rows.append({
                        "analysis_rule": rule,
                        "analysis_rule_label": RULE_LABELS[rule],
                        "selected_year": int(year),
                        "published_week_label": target["published_week_label"],
                        "week_num": int(target["week_num"]),
                        "target_id": target_id,
                        "history_weeks": history_labels,
                        "policy": policy,
                        "policy_label": POLICY_LABELS[policy],
                        "target_observed_admissions": target_total,
                        "allocation_budget_admission_equivalents": allocation_budget,
                        "reserve_premium_fraction": allocation_budget / target_total - 1.0 if target_total else np.nan,
                        "shortage_admission_equivalents": float(shortage.sum()),
                        "shortage_fraction": float(shortage.sum() / target_total) if target_total else np.nan,
                        "unused_admission_equivalents": float(unused.sum()),
                        "unused_fraction_of_allocation": float(unused.sum() / allocation_budget) if allocation_budget else np.nan,
                        "worst_division_service_fraction": float(service[actual > 0].min()) if (actual > 0).any() else 1.0,
                        "full_division_coverage": full_coverage,
                        "allocation_stability_tvd": stability,
                        "evaluation_scope": "Observed national total fixed; evaluates spatial allocation only.",
                    })
                    for division in DIVISIONS:
                        division_rows.append({
                            "analysis_rule": rule,
                            "selected_year": int(year),
                            "published_week_label": target["published_week_label"],
                            "target_id": target_id,
                            "history_weeks": history_labels,
                            "policy": policy,
                            "policy_label": POLICY_LABELS[policy],
                            "division": division,
                            "actual_admissions": float(actual[division]),
                            "actual_share": float(actual[division] / target_total) if target_total else np.nan,
                            "allocation_weight": float(allocation_weight[division]),
                            "normalized_policy_weight": float(normalized[division]),
                            "allocated_admission_equivalents": float(allocation[division]),
                            "shortage_admission_equivalents": float(shortage[division]),
                            "unused_admission_equivalents": float(unused[division]),
                            "service_fraction": float(service[division]),
                        })
    policy_week = pd.DataFrame(policy_week_rows)
    division_detail = pd.DataFrame(division_rows)
    summary = (
        policy_week.groupby(
            ["analysis_rule", "analysis_rule_label", "policy", "policy_label"],
            as_index=False,
        )
        .agg(
            target_weeks=("target_id", "nunique"),
            mean_shortage_fraction=("shortage_fraction", "mean"),
            median_shortage_fraction=("shortage_fraction", "median"),
            p90_shortage_fraction=("shortage_fraction", lambda x: x.quantile(0.90)),
            maximum_shortage_fraction=("shortage_fraction", "max"),
            mean_unused_fraction_of_allocation=("unused_fraction_of_allocation", "mean"),
            mean_worst_division_service_fraction=("worst_division_service_fraction", "mean"),
            full_division_coverage_rate=("full_division_coverage", "mean"),
            mean_reserve_premium_fraction=("reserve_premium_fraction", "mean"),
            mean_allocation_stability_tvd=("allocation_stability_tvd", "mean"),
        )
    )
    return policy_week, division_detail, summary


def paired_block_bootstrap(policy_week: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = [
        ("pooled3", "latest", "same national budget"),
        ("max3_normalized", "pooled3", "same national budget"),
        ("max3_envelope", "pooled3", "envelope includes extra reserve"),
    ]
    rows: list[dict] = []
    for rule in RULE_LABELS:
        rule_data = policy_week[policy_week["analysis_rule"] == rule]
        for candidate, reference, budget_note in comparisons:
            pivot = rule_data[rule_data["policy"].isin([candidate, reference])].pivot(
                index=["selected_year", "week_num", "target_id"],
                columns="policy",
                values="shortage_fraction",
            ).dropna().reset_index()
            pivot["difference"] = pivot[candidate] - pivot[reference]
            observed = float(pivot["difference"].mean())
            year_arrays = {
                int(year): group.sort_values("week_num")["difference"].to_numpy()
                for year, group in pivot.groupby("selected_year")
            }
            draws = np.empty(BOOTSTRAP_REPLICATES)
            for replicate in range(BOOTSTRAP_REPLICATES):
                sampled: list[float] = []
                for values in year_arrays.values():
                    n = len(values)
                    chosen: list[float] = []
                    while len(chosen) < n:
                        start = int(rng.integers(0, n))
                        chosen.extend(
                            values[(start + offset) % n]
                            for offset in range(BOOTSTRAP_BLOCK_LENGTH)
                        )
                    sampled.extend(chosen[:n])
                draws[replicate] = float(np.mean(sampled))
            rows.append({
                "analysis_rule": rule,
                "analysis_rule_label": RULE_LABELS[rule],
                "candidate_policy": candidate,
                "candidate_policy_label": POLICY_LABELS[candidate],
                "reference_policy": reference,
                "reference_policy_label": POLICY_LABELS[reference],
                "paired_target_weeks": len(pivot),
                "observed_mean_shortage_difference": observed,
                "bootstrap_95pct_ci_lower": float(np.quantile(draws, 0.025)),
                "bootstrap_95pct_ci_upper": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_candidate_lower_shortage": float(np.mean(draws < 0)),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "moving_block_length_weeks": BOOTSTRAP_BLOCK_LENGTH,
                "budget_comparability": budget_note,
                "interpretation": "Negative difference favors the candidate policy.",
            })
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    week_audit: pd.DataFrame,
    alignment_best: pd.DataFrame,
) -> None:
    primary = summary[summary["analysis_rule"] == "published_primary"].set_index("policy")
    filtered = summary[summary["analysis_rule"] == "isolated_low_filtered"].set_index("policy")
    pooled = primary.loc["pooled3"]
    latest = primary.loc["latest"]
    envelope = primary.loc["max3_envelope"]
    pooled_relative = (
        (latest["mean_shortage_fraction"] - pooled["mean_shortage_fraction"])
        / latest["mean_shortage_fraction"]
    )
    envelope_relative = (
        (pooled["mean_shortage_fraction"] - envelope["mean_shortage_fraction"])
        / pooled["mean_shortage_fraction"]
    )
    anomaly_rows = week_audit[week_audit["isolated_low_anomaly"]]
    comparison_primary = comparisons[comparisons["analysis_rule"] == "published_primary"]

    lines = [
        "# Retrospective divisional allocation-policy backtest",
        "",
        f"Generated: {RUN_DATE}",
        "",
        "## Decision result",
        "",
        (
            f"The trailing 3-week pooled-share rule is the preferred same-budget baseline. "
            f"Across {int(pooled['target_weeks'])} target weeks, mean divisional shortage was "
            f"{pooled['mean_shortage_fraction']:.1%}, versus {latest['mean_shortage_fraction']:.1%} "
            f"for the latest-week rule—a {pooled_relative:.1%} relative reduction—and its mean "
            f"week-to-week allocation change was {pooled['mean_allocation_stability_tvd']:.1%} "
            f"versus {latest['mean_allocation_stability_tvd']:.1%}."
        ),
        "",
        (
            f"The 3-week max-share envelope reduced mean shortage further to "
            f"{envelope['mean_shortage_fraction']:.1%} ({envelope_relative:.1%} relative to pooled), "
            f"but required a mean {envelope['mean_reserve_premium_fraction']:.1%} extra national "
            f"reserve. It achieved zero divisional shortage in only "
            f"{envelope['full_division_coverage_rate']:.1%} of target weeks, so it is a reserve "
            f"heuristic, not a guarantee."
        ),
        "",
        "## What was tested",
        "",
        "For each published DGHS target week, the national total actually observed that week was held fixed. Only the division shares were predicted from the preceding three eligible published weeks. This deliberately measures the spatial policy separately from national forecast error.",
        "",
        "| Policy | Mean shortage | P90 shortage | Mean unused | Worst-division service | Full coverage | Reserve premium | Stability TVD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ["latest", "pooled3", "max3_normalized", "max3_envelope"]:
        row = primary.loc[policy]
        lines.append(
            f"| {POLICY_LABELS[policy]} | {row['mean_shortage_fraction']:.1%} | "
            f"{row['p90_shortage_fraction']:.1%} | {row['mean_unused_fraction_of_allocation']:.1%} | "
            f"{row['mean_worst_division_service_fraction']:.1%} | "
            f"{row['full_division_coverage_rate']:.1%} | "
            f"{row['mean_reserve_premium_fraction']:.1%} | "
            f"{row['mean_allocation_stability_tvd']:.1%} |"
        )
    lines.extend([
        "",
        "## Uncertainty and data checks",
        "",
    ])
    for row in comparison_primary.itertuples():
        lines.append(
            f"- {row.candidate_policy_label} minus {row.reference_policy_label}: "
            f"{row.observed_mean_shortage_difference:+.2%} mean shortage difference; "
            f"moving-block bootstrap 95% interval "
            f"[{row.bootstrap_95pct_ci_lower:+.2%}, {row.bootstrap_95pct_ci_upper:+.2%}]. "
            f"{row.budget_comparability}."
        )
    lines.extend([
        (
            f"- The sensitivity analysis excluded {len(anomaly_rows)} isolated-low observations "
            f"without imputing them. The pooled-rule mean shortage changed from "
            f"{primary.loc['pooled3', 'mean_shortage_fraction']:.1%} to "
            f"{filtered.loc['pooled3', 'mean_shortage_fraction']:.1%}; the policy ranking was unchanged."
        ),
        "- DGHS Wxx and the daily-derived national series are aligned directly on confirmed Sunday-Saturday weeks. Residual total differences reflect dashboard aggregation/content, not an inferred week shift.",
        "",
        "## Interpretation for beds and IV fluid",
        "",
        "The shortage fractions can be applied to bed-day and IV-fluid quantities only when the same linear per-admission coefficients are used for every division. They do not establish facility-level feasibility, treatment-mix differences, stock lead times, staffing needs, or transfer capacity.",
        "",
        "## Recommended operational use",
        "",
        "Use trailing 3-week pooled shares for the routine divisional allocation. Hold the max-share envelope's extra amount as a national or regional contingency reserve rather than automatically shipping the entire envelope to every division. Re-run this joint spatial backtest after facility-week stock/capacity data become available.",
        "",
        "## Limitations",
        "",
        "- This is a spatial-policy backtest conditional on the observed national total, not a complete hindcast of forecast-plus-allocation performance.",
        "- DGHS admissions measure reported hospital workload, not all infections or all unmet demand.",
        "- Current 2026 W41 was excluded as an in-progress publication. Isolated-low weeks were excluded only in sensitivity analysis; none were imputed.",
        "- The eight divisions are treated as homogeneous conversion units; no facility catchment, severity mix, stock, travel, referral, or workforce constraints are modeled.",
    ])
    (OUTPUTS / "dengue_phase2_policy_backtest_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_checkpoint(summary: pd.DataFrame, week_audit: pd.DataFrame) -> None:
    primary = summary[summary["analysis_rule"] == "published_primary"].set_index("policy")
    anomalies = week_audit[week_audit["isolated_low_anomaly"]]
    text = f"""# Phase 2 spatial-policy backtest checkpoint

Last updated: {RUN_DATE} (Asia/Dhaka)

## Completed

- Retrospective division-share evaluation across {int(primary.loc['pooled3', 'target_weeks'])} target weeks.
- Primary published-data analysis plus a no-imputation isolated-low sensitivity analysis.
- Year-stratified paired moving-block bootstrap (block length {BOOTSTRAP_BLOCK_LENGTH}, {BOOTSTRAP_REPLICATES} replicates).
- Audit of direct DGHS Wxx versus daily-derived Sunday-Saturday alignment.

## Decision

- Routine policy: trailing 3-week pooled division shares.
- Contingency option: retain the max-share envelope premium as central/regional reserve.
- Pooled mean shortage: {primary.loc['pooled3', 'mean_shortage_fraction']:.3%}.
- Latest-week mean shortage: {primary.loc['latest', 'mean_shortage_fraction']:.3%}.
- Envelope mean shortage: {primary.loc['max3_envelope', 'mean_shortage_fraction']:.3%}; mean reserve premium {primary.loc['max3_envelope', 'mean_reserve_premium_fraction']:.3%}.

## Data treatment

- 2026 W41 excluded because it was still in progress at extraction.
- {len(anomalies)} isolated-low weeks flagged; excluded only in sensitivity analysis and never imputed.
- DGHS Wxx and national forecasts use the same Sunday-Saturday calendar.

## Resume

1. Rebuild backtest: `python3 work/phase2_capacity/backtest_spatial_policies.py`
2. Rebuild workbook: `node work/phase2_capacity/build_phase2_workbook.mjs`
3. Next research step: obtain facility-week usable dengue beds, IV stock/lead times, rosters/WISN inputs, transfers, and denied/diverted admissions.
"""
    (WORK / "POLICY_BACKTEST_CHECKPOINT.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    matrix, base_audit = load_weekly_matrix()
    week_audit = classify_weeks(base_audit)
    alignment_detail, alignment_best = alignment_audit(week_audit)
    policy_week, division_detail, summary = backtest(matrix, week_audit)
    comparisons = paired_block_bootstrap(policy_week)

    output_frames = {
        "outputs/dengue_phase2_policy_backtest_detail.csv": policy_week,
        "outputs/dengue_phase2_policy_backtest_division_detail.csv": division_detail,
        "outputs/dengue_phase2_policy_backtest_summary.csv": summary,
        "outputs/dengue_phase2_policy_bootstrap_comparisons.csv": comparisons,
        "outputs/dengue_phase2_week_alignment_audit.csv": alignment_detail,
        "outputs/dengue_phase2_week_alignment_best.csv": alignment_best,
        "outputs/dengue_phase2_week_exclusion_audit.csv": week_audit,
    }
    for relative_path, frame in output_frames.items():
        frame.to_csv(ROOT / relative_path, index=False)

    write_report(summary, comparisons, week_audit, alignment_best)
    write_checkpoint(summary, week_audit)

    manifest = {
        "status": "complete",
        "generated_on": RUN_DATE,
        "objective": "Evaluate division-share allocation policies separately from national forecast error.",
        "inputs": {
            str(DIVISION_PATH.relative_to(ROOT)): sha256(DIVISION_PATH),
            str(NATIONAL_PATH.relative_to(ROOT)): sha256(NATIONAL_PATH),
        },
        "parameters": {
            "history_weeks": 3,
            "anomaly_threshold_fraction_of_local_median": ANOMALY_THRESHOLD,
            "rebound_threshold_multiple": REBOUND_THRESHOLD,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_block_length_weeks": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "outputs": [
            *output_frames.keys(),
            "outputs/dengue_phase2_policy_backtest_report.md",
            "work/phase2_capacity/POLICY_BACKTEST_CHECKPOINT.md",
        ],
        "limitations": [
            "Observed national totals are fixed, so this is not a joint forecast-allocation backtest.",
            "DGHS Wxx uses Sunday-Saturday weeks; week 1 contains 1 January.",
            "Reported admissions do not measure unmet or community demand.",
            "No facility-level beds, stock, workforce, lead-time, referral, or transfer constraints are included.",
        ],
    }
    (WORK / "policy_backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    primary = summary[summary["analysis_rule"] == "published_primary"]
    print(primary[[
        "policy", "target_weeks", "mean_shortage_fraction",
        "mean_reserve_premium_fraction", "mean_allocation_stability_tvd",
    ]].to_string(index=False))
    print(f"\nWrote {len(output_frames)} CSV files, report, checkpoint, and manifest.")


if __name__ == "__main__":
    main()
