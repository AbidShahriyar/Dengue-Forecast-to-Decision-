#!/usr/bin/env python3
"""Joint one-week national-forecast and divisional-allocation hindcast.

The selected one-week biology-informed national forecast is combined with
division shares estimated from the preceding three eligible DGHS-published
weeks. DGHS Wxx is mapped directly to the confirmed Sunday-Saturday calendar;
week 1 is the interval containing 1 January.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase2_capacity"
OUTPUTS = ROOT / "outputs"
DIVISION_PATH = ROOT / "work/weekly_covariates/dengue_weekly_division_published.csv"
NATIONAL_PATH = ROOT / "work/weekly_covariates/dengue_weekly_national.csv"
PREDICTION_PATH = OUTPUTS / "dengue_biology_backtest_predictions.csv"
SELECTED_MODEL_PATH = OUTPUTS / "dengue_biology_selected_models.csv"
WEEK_AUDIT_PATH = OUTPUTS / "dengue_phase2_week_exclusion_audit.csv"
RUN_DATE = "2026-10-14"

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
FORECAST_LEVEL_LABELS = {
    "observed_total_oracle": "Observed national total (diagnostic oracle)",
    "point": "Point forecast",
    "upper80": "Upper 80% conformal bound",
}
STRATEGY_LABELS = {
    "observed_total_oracle__pooled3": "Observed-total oracle + pooled shares",
    "point__latest": "Point forecast + latest share",
    "point__pooled3": "Point forecast + pooled shares",
    "point__max3_normalized": "Point forecast + normalized max shares",
    "point__max3_envelope": "Point forecast + max-share envelope",
    "upper80__pooled3": "Upper 80% bound + pooled shares",
    "upper80__max3_envelope": "Upper 80% bound + max-share envelope",
}
ALIGNMENT_LABELS = {
    "confirmed_dghs": "Confirmed DGHS Sunday-Saturday week",
}
RULE_LABELS = {
    "published_primary": "Published-data primary analysis",
    "isolated_low_filtered": "Isolated-low sensitivity analysis",
}
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 4
BOOTSTRAP_SEED = 20261013
RESERVE_UPLIFTS = [value / 100 for value in range(0, 101, 5)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dghs_week_fields(day: date) -> tuple[int, int]:
    start = day - timedelta(days=(day.weekday() + 1) % 7)
    end = start + timedelta(days=6)
    week_year = end.year if end.year != start.year else start.year
    anchor = date(week_year, 1, 1)
    first_start = anchor - timedelta(days=(anchor.weekday() + 1) % 7)
    week_number = ((start - first_start).days // 7) + 1
    return week_year, week_number


def load_spatial_matrix() -> pd.DataFrame:
    raw = pd.read_csv(DIVISION_PATH)
    raw = raw[raw["published_week_label"].notna()].copy()
    raw["week_num"] = raw["published_week_label"].str.extract(r"(\d+)").astype(int)
    matrix = raw.pivot_table(
        index=["selected_year", "published_week_label", "week_num"],
        columns="division_label",
        values="admissions",
        aggfunc="sum",
    ).reset_index()
    for division in DIVISIONS:
        if division not in matrix.columns:
            matrix[division] = np.nan
    return matrix[["selected_year", "published_week_label", "week_num", *DIVISIONS]].sort_values(
        ["selected_year", "week_num"]
    ).reset_index(drop=True)


def load_selected_predictions() -> tuple[pd.DataFrame, str]:
    selected = pd.read_csv(SELECTED_MODEL_PATH)
    selected_model = selected.loc[selected["horizon"].eq(1), "selected_model"].iloc[0]
    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["target_week_start"])
    predictions = predictions[
        predictions["horizon"].eq(1)
        & predictions["model"].eq(selected_model)
        & predictions["lower_80_conformal"].notna()
        & predictions["upper_80_conformal"].notna()
    ].copy()
    fields = predictions["target_week_start"].dt.date.map(dghs_week_fields)
    predictions[["week_year", "week_number"]] = pd.DataFrame(
        fields.tolist(), index=predictions.index
    )
    if predictions.duplicated(["week_year", "week_number"]).any():
        raise RuntimeError("Selected forecast has duplicate DGHS target weeks.")
    return predictions, selected_model


def eligible_matrix(
    matrix: pd.DataFrame,
    week_audit: pd.DataFrame,
    data_rule: str,
) -> pd.DataFrame:
    include_column = {
        "published_primary": "included_published_primary",
        "isolated_low_filtered": "included_isolated_low_filtered",
    }[data_rule]
    keys = set(
        zip(
            week_audit.loc[week_audit[include_column], "selected_year"],
            week_audit.loc[week_audit[include_column], "published_week_label"],
        )
    )
    keep = matrix.apply(
        lambda row: (row["selected_year"], row["published_week_label"]) in keys,
        axis=1,
    )
    return matrix[keep].copy()


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
    allocation_weight = raw if policy == "max3_envelope" else normalized
    return allocation_weight.astype(float), normalized.astype(float)


def allocation_result(
    actual: pd.Series,
    allocation_weight: pd.Series,
    normalized_weight: pd.Series,
    national_amount: float,
) -> dict:
    actual_total = float(actual.sum())
    allocation = allocation_weight * national_amount
    shortage = (actual - allocation).clip(lower=0)
    unused = (allocation - actual).clip(lower=0)
    allocation_total = float(allocation.sum())
    service = pd.Series(
        np.where(actual > 0, np.minimum(allocation / actual, 1.0), 1.0),
        index=DIVISIONS,
    )
    return {
        "allocation": allocation,
        "shortage": shortage,
        "unused": unused,
        "service": service,
        "actual_total": actual_total,
        "allocation_total": allocation_total,
        "shortage_total": float(shortage.sum()),
        "unused_total": float(unused.sum()),
        "shortage_fraction": float(shortage.sum() / actual_total) if actual_total else np.nan,
        "unused_fraction": float(unused.sum() / allocation_total) if allocation_total else np.nan,
        "budget_to_actual_ratio": allocation_total / actual_total if actual_total else np.nan,
        "worst_service": float(service[actual > 0].min()) if (actual > 0).any() else 1.0,
        "full_coverage": bool((shortage <= 1e-9).all()),
        "normalized_weight": normalized_weight,
    }


def run_joint_backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    matrix = load_spatial_matrix()
    week_audit = pd.read_csv(WEEK_AUDIT_PATH)
    predictions, selected_model = load_selected_predictions()
    prediction_lookup = {
        (int(row.week_year), int(row.week_number)): row
        for row in predictions.itertuples()
    }
    joint_rows: list[dict] = []
    division_rows: list[dict] = []
    alignment_rows: list[dict] = []
    for data_rule in RULE_LABELS:
        eligible = eligible_matrix(matrix, week_audit, data_rule)
        for year, year_data in eligible.groupby("selected_year", sort=True):
            year = int(year)
            year_data = year_data.sort_values("week_num").reset_index(drop=True)
            for position in range(3, len(year_data)):
                target = year_data.iloc[position]
                history = year_data.iloc[position - 3:position]
                actual = target[DIVISIONS].astype(float)
                actual_total = float(actual.sum())
                alignment = "confirmed_dghs"
                week_number = int(target["week_num"])
                alignment_rows.append({
                    "analysis_rule": data_rule,
                    "selected_year": year,
                    "published_week_label": target["published_week_label"],
                    "published_week_number": week_number,
                    "alignment_scenario": alignment,
                    "week_definition": "Sunday-Saturday; week 1 contains 1 January",
                    "mapping_shift": 0,
                })
                for alignment in ALIGNMENT_LABELS:
                    prediction = prediction_lookup.get((year, week_number))
                    if prediction is None:
                        continue
                    lower = float(prediction.lower_80_conformal)
                    upper = float(prediction.upper_80_conformal)
                    point = float(prediction.prediction)
                    mapped_daily_actual = float(prediction.actual)
                    forecast_levels = {
                        "observed_total_oracle": actual_total,
                        "point": point,
                        "upper80": upper,
                    }
                    for forecast_level, national_amount in forecast_levels.items():
                        for policy in POLICY_LABELS:
                            allocation_weight, normalized_weight = policy_weights(history, policy)
                            result = allocation_result(
                                actual, allocation_weight, normalized_weight, national_amount
                            )
                            target_id = f"{year}-{target['published_week_label']}"
                            strategy_id = f"{forecast_level}__{policy}"
                            joint_rows.append({
                                "analysis_rule": data_rule,
                                "analysis_rule_label": RULE_LABELS[data_rule],
                                "alignment_scenario": alignment,
                                "alignment_label": ALIGNMENT_LABELS[alignment],
                                "selected_year": year,
                                "published_week_label": target["published_week_label"],
                                "published_week_num": int(target["week_num"]),
                                "mapped_week_number": week_number,
                                "mapping_shift": 0,
                                "target_id": target_id,
                                "history_weeks": "|".join(history["published_week_label"].astype(str)),
                                "forecast_model": selected_model,
                                "forecast_level": forecast_level,
                                "forecast_level_label": FORECAST_LEVEL_LABELS[forecast_level],
                                "policy": policy,
                                "policy_label": POLICY_LABELS[policy],
                                "strategy_id": strategy_id,
                                "mapped_daily_actual_admissions": mapped_daily_actual,
                                "division_actual_admissions": actual_total,
                                "division_minus_daily_actual": actual_total - mapped_daily_actual,
                                "division_minus_daily_fraction": (
                                    (actual_total - mapped_daily_actual) / mapped_daily_actual
                                    if mapped_daily_actual else np.nan
                                ),
                                "national_point_forecast": point,
                                "national_lower_80_conformal": lower,
                                "national_upper_80_conformal": upper,
                                "national_amount_used": national_amount,
                                "point_forecast_error_vs_division_total": point - actual_total,
                                "point_absolute_error_vs_division_total": abs(point - actual_total),
                                "mapped_interval_covers_division_total": lower <= actual_total <= upper,
                                "mapped_interval_covers_daily_actual": lower <= mapped_daily_actual <= upper,
                                "allocation_budget_admission_equivalents": result["allocation_total"],
                                "allocation_budget_to_actual_ratio": result["budget_to_actual_ratio"],
                                "policy_spatial_reserve_fraction": float(allocation_weight.sum() - 1.0),
                                "shortage_admission_equivalents": result["shortage_total"],
                                "shortage_fraction": result["shortage_fraction"],
                                "unused_admission_equivalents": result["unused_total"],
                                "unused_fraction_of_allocation": result["unused_fraction"],
                                "worst_division_service_fraction": result["worst_service"],
                                "full_division_coverage": result["full_coverage"],
                            })
                            for division in DIVISIONS:
                                division_rows.append({
                                    "analysis_rule": data_rule,
                                    "alignment_scenario": alignment,
                                    "selected_year": year,
                                    "published_week_label": target["published_week_label"],
                                    "mapped_week_number": week_number,
                                    "mapping_shift": 0,
                                    "target_id": target_id,
                                    "history_weeks": "|".join(history["published_week_label"].astype(str)),
                                    "forecast_level": forecast_level,
                                    "policy": policy,
                                    "division": division,
                                    "actual_admissions": float(actual[division]),
                                    "allocation_weight": float(allocation_weight[division]),
                                    "normalized_policy_weight": float(normalized_weight[division]),
                                    "allocated_admission_equivalents": float(result["allocation"][division]),
                                    "shortage_admission_equivalents": float(result["shortage"][division]),
                                    "unused_admission_equivalents": float(result["unused"][division]),
                                    "service_fraction": float(result["service"][division]),
                                })
    return (
        pd.DataFrame(joint_rows),
        pd.DataFrame(division_rows),
        pd.DataFrame(alignment_rows),
        week_audit,
        selected_model,
    )


def summarize(joint: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "analysis_rule", "analysis_rule_label", "alignment_scenario", "alignment_label",
        "forecast_level", "forecast_level_label", "policy", "policy_label",
    ]

    def group_summary(group: pd.DataFrame) -> pd.Series:
        actual_total = group["division_actual_admissions"].sum()
        allocation_total = group["allocation_budget_admission_equivalents"].sum()
        shortage_total = group["shortage_admission_equivalents"].sum()
        point_absolute = group["point_absolute_error_vs_division_total"].sum()
        point_bias = group["point_forecast_error_vs_division_total"].sum()
        return pd.Series({
            "target_weeks": group["target_id"].nunique(),
            "mean_shortage_fraction": group["shortage_fraction"].mean(),
            "median_shortage_fraction": group["shortage_fraction"].median(),
            "p90_shortage_fraction": group["shortage_fraction"].quantile(0.90),
            "mean_unused_fraction_of_allocation": group["unused_fraction_of_allocation"].mean(),
            "mean_budget_to_actual_ratio": group["allocation_budget_to_actual_ratio"].mean(),
            "aggregate_budget_to_actual_ratio": allocation_total / actual_total,
            "total_actual_admissions": actual_total,
            "total_shortage_admission_equivalents": shortage_total,
            "mean_weekly_shortage_admission_equivalents": group[
                "shortage_admission_equivalents"
            ].mean(),
            "demand_weighted_shortage_fraction": shortage_total / actual_total,
            "median_budget_to_actual_ratio": group["allocation_budget_to_actual_ratio"].median(),
            "mean_worst_division_service_fraction": group["worst_division_service_fraction"].mean(),
            "full_division_coverage_rate": group["full_division_coverage"].mean(),
            "point_forecast_wape_vs_division_total": point_absolute / actual_total if actual_total else np.nan,
            "point_forecast_bias_fraction_vs_division_total": point_bias / actual_total if actual_total else np.nan,
            "mapped_interval_coverage_vs_division_total": group["mapped_interval_covers_division_total"].mean(),
            "mapped_interval_coverage_vs_daily_actual": group["mapped_interval_covers_daily_actual"].mean(),
            "mean_absolute_division_daily_aggregation_gap_fraction": group["division_minus_daily_fraction"].abs().mean(),
        })

    return joint.groupby(keys, as_index=False).apply(
        group_summary, include_groups=False
    ).reset_index(drop=True)


def reserve_frontier(joint: pd.DataFrame) -> pd.DataFrame:
    base = joint[
        joint["analysis_rule"].eq("published_primary")
        & joint["alignment_scenario"].eq("confirmed_dghs")
        & joint["forecast_level"].eq("point")
        & joint["policy"].eq("pooled3")
    ].copy()
    rows: list[dict] = []
    for row in base.itertuples():
        actual = row.division_actual_admissions
        point = row.national_point_forecast
        for uplift in RESERVE_UPLIFTS:
            # Scaling a pooled-share allocation preserves division weights.
            base_shortage = None
            budget = row.allocation_budget_admission_equivalents * (1.0 + uplift)
            # The division detail is not needed here: reconstruct each division from the
            # joint division file would make this function order-dependent. The caller
            # supplies the exact recomputation below through a keyed extract.
            rows.append({
                "target_id": row.target_id,
                "selected_year": row.selected_year,
                "published_week_label": row.published_week_label,
                "reserve_rule": f"point_plus_{int(round(uplift * 100))}pct",
                "reserve_rule_label": f"Point forecast + {int(round(uplift * 100))}%",
                "uplift_fraction": uplift,
                "actual_total": actual,
                "point_forecast": point,
                "allocation_budget": budget,
            })
    return pd.DataFrame(rows)


def complete_reserve_frontier(
    reserve_rows: pd.DataFrame,
    division_detail: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    division_base = division_detail[
        division_detail["analysis_rule"].eq("published_primary")
        & division_detail["alignment_scenario"].eq("confirmed_dghs")
        & division_detail["forecast_level"].eq("point")
        & division_detail["policy"].eq("pooled3")
    ].copy()
    detail_lookup = {
        target_id: group
        for target_id, group in division_base.groupby("target_id")
    }
    target_totals = division_base.groupby("target_id")["actual_admissions"].sum()
    surge_threshold = float(target_totals.quantile(0.75))
    evaluated: list[dict] = []
    for row in reserve_rows.itertuples():
        detail = detail_lookup[row.target_id]
        actual = detail.set_index("division")["actual_admissions"].reindex(DIVISIONS)
        base_allocation = detail.set_index("division")["allocated_admission_equivalents"].reindex(DIVISIONS)
        allocation = base_allocation * (1.0 + row.uplift_fraction)
        shortage = (actual - allocation).clip(lower=0)
        unused = (allocation - actual).clip(lower=0)
        service = np.where(actual > 0, np.minimum(allocation / actual, 1.0), 1.0)
        evaluated.append({
            **row._asdict(),
            "shortage_total": float(shortage.sum()),
            "unused_total": float(unused.sum()),
            "shortage_fraction": float(shortage.sum() / actual.sum()),
            "unused_fraction_of_allocation": float(unused.sum() / allocation.sum()),
            "budget_to_actual_ratio": float(allocation.sum() / actual.sum()),
            "worst_division_service_fraction": float(np.min(service[actual.to_numpy() > 0])),
            "full_division_coverage": bool((shortage <= 1e-9).all()),
            "surge_week": bool(actual.sum() >= surge_threshold),
        })
    evaluated_frame = pd.DataFrame(evaluated)

    # Add adaptive reserve rules for direct comparison with the fixed uplift grid.
    adaptive = division_detail[
        division_detail["analysis_rule"].eq("published_primary")
        & division_detail["alignment_scenario"].eq("confirmed_dghs")
        & (
            (division_detail["forecast_level"].eq("upper80") & division_detail["policy"].eq("pooled3"))
            | (division_detail["forecast_level"].eq("point") & division_detail["policy"].eq("max3_envelope"))
            | (division_detail["forecast_level"].eq("upper80") & division_detail["policy"].eq("max3_envelope"))
        )
    ].copy()
    adaptive_rows: list[dict] = []
    for (forecast_level, policy, target_id), group in adaptive.groupby(
        ["forecast_level", "policy", "target_id"]
    ):
        actual = group["actual_admissions"].sum()
        allocation = group["allocated_admission_equivalents"].sum()
        shortage = group["shortage_admission_equivalents"].sum()
        unused = group["unused_admission_equivalents"].sum()
        if forecast_level == "upper80" and policy == "pooled3":
            label = "Upper 80% bound with pooled shares"
            rule = "upper80_pooled"
        elif forecast_level == "upper80" and policy == "max3_envelope":
            label = "Upper 80% bound with max-share envelope"
            rule = "upper80_max3_envelope"
        else:
            label = "Point forecast with max-share envelope"
            rule = "point_max3_envelope"
        first = group.iloc[0]
        adaptive_rows.append({
            "target_id": target_id,
            "selected_year": first["selected_year"],
            "published_week_label": first["published_week_label"],
            "reserve_rule": rule,
            "reserve_rule_label": label,
            "uplift_fraction": np.nan,
            "actual_total": actual,
            "point_forecast": np.nan,
            "allocation_budget": allocation,
            "shortage_total": shortage,
            "unused_total": unused,
            "shortage_fraction": shortage / actual if actual else np.nan,
            "unused_fraction_of_allocation": unused / allocation if allocation else np.nan,
            "budget_to_actual_ratio": allocation / actual if actual else np.nan,
            "worst_division_service_fraction": group["service_fraction"].min(),
            "full_division_coverage": bool((group["shortage_admission_equivalents"] <= 1e-9).all()),
            "surge_week": bool(actual >= surge_threshold),
        })
    adaptive_frame = pd.DataFrame(adaptive_rows)

    # Exact fixed-reserve comparators with the same ratio of total allocated
    # units to total observed admissions as each adaptive policy.
    base_rows = evaluated_frame.loc[
        evaluated_frame["reserve_rule"].eq("point_plus_0pct")
    ]
    base_budget_ratio = float(
        base_rows["allocation_budget"].sum() / base_rows["actual_total"].sum()
    )
    matched_rows: list[dict] = []
    for adaptive_rule, adaptive_group in adaptive_frame.groupby("reserve_rule"):
        target_budget_ratio = float(
            adaptive_group["allocation_budget"].sum()
            / adaptive_group["actual_total"].sum()
        )
        matched_uplift = target_budget_ratio / base_budget_ratio - 1.0
        adaptive_label = str(adaptive_group["reserve_rule_label"].iloc[0])
        for target_id, detail in detail_lookup.items():
            actual = detail.set_index("division")["actual_admissions"].reindex(DIVISIONS)
            base_allocation = detail.set_index("division")[
                "allocated_admission_equivalents"
            ].reindex(DIVISIONS)
            allocation = base_allocation * (1.0 + matched_uplift)
            shortage = (actual - allocation).clip(lower=0)
            unused = (allocation - actual).clip(lower=0)
            service = np.where(actual > 0, np.minimum(allocation / actual, 1.0), 1.0)
            first = detail.iloc[0]
            matched_rows.append({
                "target_id": target_id,
                "selected_year": first["selected_year"],
                "published_week_label": first["published_week_label"],
                "reserve_rule": f"matched_fixed_for_{adaptive_rule}",
                "reserve_rule_label": f"Budget-matched fixed reserve for {adaptive_label}",
                "uplift_fraction": matched_uplift,
                "actual_total": float(actual.sum()),
                "point_forecast": float(base_allocation.sum()),
                "allocation_budget": float(allocation.sum()),
                "shortage_total": float(shortage.sum()),
                "unused_total": float(unused.sum()),
                "shortage_fraction": float(shortage.sum() / actual.sum()),
                "unused_fraction_of_allocation": float(unused.sum() / allocation.sum()),
                "budget_to_actual_ratio": float(allocation.sum() / actual.sum()),
                "worst_division_service_fraction": float(
                    np.min(service[actual.to_numpy() > 0])
                ),
                "full_division_coverage": bool((shortage <= 1e-9).all()),
                "surge_week": bool(actual.sum() >= surge_threshold),
                "matched_adaptive_rule": adaptive_rule,
            })
    adaptive_frame["matched_adaptive_rule"] = adaptive_frame["reserve_rule"]
    evaluated_frame["matched_adaptive_rule"] = ""
    all_rows = pd.concat(
        [evaluated_frame, adaptive_frame, pd.DataFrame(matched_rows)],
        ignore_index=True,
    )
    order = {
        **{
            f"point_plus_{int(round(u * 100))}pct": index
            for index, u in enumerate(RESERVE_UPLIFTS)
        },
        "point_max3_envelope": 30,
        "matched_fixed_for_point_max3_envelope": 31,
        "upper80_pooled": 32,
        "matched_fixed_for_upper80_pooled": 33,
        "upper80_max3_envelope": 34,
        "matched_fixed_for_upper80_max3_envelope": 35,
    }

    def summarize_rule(group: pd.DataFrame) -> pd.Series:
        surge = group[group["surge_week"]]
        nonsurge = group[~group["surge_week"]]
        return pd.Series({
            "target_weeks": group["target_id"].nunique(),
            "uplift_fraction": (
                group["uplift_fraction"].dropna().iloc[0]
                if group["uplift_fraction"].notna().any() else np.nan
            ),
            "mean_shortage_fraction": group["shortage_fraction"].mean(),
            "demand_weighted_shortage_fraction": (
                group["shortage_total"].sum() / group["actual_total"].sum()
            ),
            "surge_mean_shortage_fraction": surge["shortage_fraction"].mean(),
            "nonsurge_mean_shortage_fraction": nonsurge["shortage_fraction"].mean(),
            "p90_shortage_fraction": group["shortage_fraction"].quantile(0.90),
            "mean_unused_fraction_of_allocation": group[
                "unused_fraction_of_allocation"
            ].mean(),
            "allocation_weighted_unused_fraction": (
                group["unused_total"].sum() / group["allocation_budget"].sum()
            ),
            "mean_budget_to_actual_ratio": group["budget_to_actual_ratio"].mean(),
            "aggregate_budget_to_actual_ratio": (
                group["allocation_budget"].sum() / group["actual_total"].sum()
            ),
            "total_actual_admissions": group["actual_total"].sum(),
            "total_shortage_admission_equivalents": group["shortage_total"].sum(),
            "mean_weekly_shortage_admission_equivalents": group[
                "shortage_total"
            ].mean(),
            "mean_worst_division_service_fraction": group[
                "worst_division_service_fraction"
            ].mean(),
            "full_division_coverage_rate": group["full_division_coverage"].mean(),
        })

    summary = all_rows.groupby(
        ["reserve_rule", "reserve_rule_label"], as_index=False
    ).apply(summarize_rule, include_groups=False).reset_index(drop=True)
    summary["display_order"] = summary["reserve_rule"].map(order)
    return (
        summary.sort_values("display_order").reset_index(drop=True),
        all_rows.sort_values(["reserve_rule", "target_id"]).reset_index(drop=True),
    )


def budget_matched_comparisons(frontier_detail: pd.DataFrame) -> pd.DataFrame:
    """Compare adaptive policies with exact aggregate-budget-matched reserves."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 29)
    adaptive_rules = [
        "point_max3_envelope", "upper80_pooled", "upper80_max3_envelope",
    ]
    rows: list[dict] = []
    for adaptive_rule in adaptive_rules:
        fixed_rule = f"matched_fixed_for_{adaptive_rule}"
        subset = frontier_detail[
            frontier_detail["reserve_rule"].isin([adaptive_rule, fixed_rule])
        ].copy()
        wide = subset.pivot(
            index=["selected_year", "target_id", "surge_week", "actual_total"],
            columns="reserve_rule",
            values=[
                "shortage_fraction", "shortage_total",
                "unused_fraction_of_allocation", "budget_to_actual_ratio",
            ],
        ).dropna().reset_index()
        shortage_difference = (
            wide[("shortage_fraction", adaptive_rule)]
            - wide[("shortage_fraction", fixed_rule)]
        )
        shortage_total_difference = (
            wide[("shortage_total", adaptive_rule)]
            - wide[("shortage_total", fixed_rule)]
        )
        unused_difference = (
            wide[("unused_fraction_of_allocation", adaptive_rule)]
            - wide[("unused_fraction_of_allocation", fixed_rule)]
        )
        adaptive_budget = wide[("budget_to_actual_ratio", adaptive_rule)] * wide[("actual_total", "")]
        fixed_budget = wide[("budget_to_actual_ratio", fixed_rule)] * wide[("actual_total", "")]
        budget_difference = adaptive_budget - fixed_budget
        analysis = pd.DataFrame({
            "selected_year": wide[("selected_year", "")].astype(int),
            "target_id": wide[("target_id", "")],
            "surge_week": wide[("surge_week", "")].astype(bool),
            "actual_total": wide[("actual_total", "")].astype(float),
            "shortage_difference": shortage_difference.astype(float),
            "shortage_total_difference": shortage_total_difference.astype(float),
            "unused_difference": unused_difference.astype(float),
            "budget_difference": budget_difference.astype(float),
        })
        year_arrays = {
            int(year): group.sort_values("target_id")[
                ["shortage_difference", "shortage_total_difference", "actual_total", "surge_week"]
            ].to_numpy()
            for year, group in analysis.groupby("selected_year", sort=True)
        }
        mean_draws = np.empty(BOOTSTRAP_REPLICATES)
        weighted_draws = np.empty(BOOTSTRAP_REPLICATES)
        surge_draws = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled: list[np.ndarray] = []
            for values in year_arrays.values():
                n = len(values)
                chosen: list[np.ndarray] = []
                while len(chosen) < n:
                    start = int(rng.integers(0, n))
                    chosen.extend(
                        values[(start + offset) % n]
                        for offset in range(BOOTSTRAP_BLOCK_LENGTH)
                    )
                sampled.extend(chosen[:n])
            draw = np.asarray(sampled)
            mean_draws[replicate] = draw[:, 0].mean()
            weighted_draws[replicate] = draw[:, 1].sum() / draw[:, 2].sum()
            surge_draws[replicate] = draw[draw[:, 3] > 0.5, 0].mean()
        adaptive_label = str(
            subset.loc[
                subset["reserve_rule"].eq(adaptive_rule), "reserve_rule_label"
            ].iloc[0]
        )
        matched_uplift = float(
            subset.loc[
                subset["reserve_rule"].eq(fixed_rule), "uplift_fraction"
            ].iloc[0]
        )
        observed_mean = float(analysis["shortage_difference"].mean())
        observed_weighted = float(
            analysis["shortage_total_difference"].sum()
            / analysis["actual_total"].sum()
        )
        observed_budget = float(
            analysis["budget_difference"].sum() / analysis["actual_total"].sum()
        )
        rows.append({
            "adaptive_rule": adaptive_rule,
            "adaptive_rule_label": adaptive_label,
            "matched_fixed_rule": fixed_rule,
            "matched_fixed_uplift_fraction": matched_uplift,
            "paired_target_weeks": int(len(analysis)),
            "adaptive_minus_fixed_mean_shortage": observed_mean,
            "mean_shortage_difference_95pct_ci_lower": float(
                np.quantile(mean_draws, 0.025)
            ),
            "mean_shortage_difference_95pct_ci_upper": float(
                np.quantile(mean_draws, 0.975)
            ),
            "probability_adaptive_lower_mean_shortage": float(np.mean(mean_draws < 0)),
            "adaptive_minus_fixed_demand_weighted_shortage": observed_weighted,
            "weighted_shortage_difference_95pct_ci_lower": float(
                np.quantile(weighted_draws, 0.025)
            ),
            "weighted_shortage_difference_95pct_ci_upper": float(
                np.quantile(weighted_draws, 0.975)
            ),
            "adaptive_minus_fixed_surge_mean_shortage": float(
                analysis.loc[analysis["surge_week"], "shortage_difference"].mean()
            ),
            "surge_shortage_difference_95pct_ci_lower": float(
                np.quantile(surge_draws, 0.025)
            ),
            "surge_shortage_difference_95pct_ci_upper": float(
                np.quantile(surge_draws, 0.975)
            ),
            "adaptive_minus_fixed_nonsurge_mean_shortage": float(
                analysis.loc[~analysis["surge_week"], "shortage_difference"].mean()
            ),
            "adaptive_minus_fixed_mean_unused": float(analysis["unused_difference"].mean()),
            "adaptive_minus_fixed_aggregate_budget_ratio": observed_budget,
            "adaptive_dominated_on_mean_shortage_and_budget": bool(
                observed_mean > 0 and observed_budget >= -1e-10
            ),
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "moving_block_length_weeks": BOOTSTRAP_BLOCK_LENGTH,
        })
    return pd.DataFrame(rows)


def paired_bootstrap(joint: pd.DataFrame) -> pd.DataFrame:
    primary = joint[
        joint["analysis_rule"].eq("published_primary")
        & joint["alignment_scenario"].eq("confirmed_dghs")
    ].copy()
    comparisons = [
        ("point__pooled3", "point__latest", "same national forecast budget"),
        ("point__max3_normalized", "point__pooled3", "same national forecast budget"),
        ("point__max3_envelope", "point__pooled3", "adds a spatial reserve envelope"),
        ("upper80__pooled3", "point__pooled3", "uses the upper conformal national forecast"),
        ("upper80__max3_envelope", "point__max3_envelope", "adds forecast-interval reserve"),
        ("point__pooled3", "observed_total_oracle__pooled3", "shows forecast plus aggregation effect"),
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict] = []
    for candidate, reference, note in comparisons:
        subset = primary[primary["strategy_id"].isin([candidate, reference])]
        shortage = subset.pivot(
            index=["selected_year", "published_week_num", "target_id"],
            columns="strategy_id",
            values="shortage_fraction",
        ).dropna().reset_index()
        budget = subset.pivot(
            index=["selected_year", "published_week_num", "target_id"],
            columns="strategy_id",
            values="allocation_budget_to_actual_ratio",
        ).dropna().reset_index()
        paired = shortage.merge(
            budget,
            on=["selected_year", "published_week_num", "target_id"],
            suffixes=("_shortage", "_budget"),
        )
        paired["shortage_difference"] = paired[f"{candidate}_shortage"] - paired[f"{reference}_shortage"]
        paired["budget_difference"] = paired[f"{candidate}_budget"] - paired[f"{reference}_budget"]
        observed_shortage = float(paired["shortage_difference"].mean())
        observed_budget = float(paired["budget_difference"].mean())
        year_arrays = {
            int(year): group.sort_values("published_week_num")[["shortage_difference", "budget_difference"]].to_numpy()
            for year, group in paired.groupby("selected_year")
        }
        shortage_draws = np.empty(BOOTSTRAP_REPLICATES)
        budget_draws = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            sampled: list[np.ndarray] = []
            for values in year_arrays.values():
                n = len(values)
                chosen: list[np.ndarray] = []
                while len(chosen) < n:
                    start = int(rng.integers(0, n))
                    chosen.extend(
                        values[(start + offset) % n]
                        for offset in range(BOOTSTRAP_BLOCK_LENGTH)
                    )
                sampled.extend(chosen[:n])
            sampled_array = np.asarray(sampled)
            shortage_draws[replicate] = sampled_array[:, 0].mean()
            budget_draws[replicate] = sampled_array[:, 1].mean()
        rows.append({
            "candidate_strategy": candidate,
            "candidate_strategy_label": STRATEGY_LABELS[candidate],
            "reference_strategy": reference,
            "reference_strategy_label": STRATEGY_LABELS[reference],
            "paired_target_weeks": len(paired),
            "observed_mean_shortage_difference": observed_shortage,
            "shortage_difference_95pct_ci_lower": float(np.quantile(shortage_draws, 0.025)),
            "shortage_difference_95pct_ci_upper": float(np.quantile(shortage_draws, 0.975)),
            "probability_candidate_lower_shortage": float(np.mean(shortage_draws < 0)),
            "observed_mean_budget_ratio_difference": observed_budget,
            "budget_difference_95pct_ci_lower": float(np.quantile(budget_draws, 0.025)),
            "budget_difference_95pct_ci_upper": float(np.quantile(budget_draws, 0.975)),
            "comparison_note": note,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "moving_block_length_weeks": BOOTSTRAP_BLOCK_LENGTH,
        })
    return pd.DataFrame(rows)


def direct_allocation_result(actual: pd.Series, allocation: pd.Series) -> dict:
    """Evaluate a vector of division-specific forecasts as allocations."""
    allocation = allocation.reindex(DIVISIONS).clip(lower=0).astype(float)
    actual = actual.reindex(DIVISIONS).astype(float)
    shortage = (actual - allocation).clip(lower=0)
    unused = (allocation - actual).clip(lower=0)
    service = pd.Series(
        np.where(actual > 0, np.minimum(allocation / actual, 1.0), 1.0),
        index=DIVISIONS,
    )
    return {
        "allocation_total": float(allocation.sum()),
        "shortage_total": float(shortage.sum()),
        "unused_total": float(unused.sum()),
        "shortage_fraction": float(shortage.sum() / actual.sum()),
        "unused_fraction": float(unused.sum() / allocation.sum()) if allocation.sum() else np.nan,
        "budget_to_actual_ratio": float(allocation.sum() / actual.sum()),
        "worst_service": float(service[actual > 0].min()) if (actual > 0).any() else 1.0,
        "full_coverage": bool((shortage <= 1e-9).all()),
    }


def allocation_forecast_sensitivity(
    joint: pd.DataFrame,
    selected_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat point-allocation evaluation under prospective forecast choices.

    The target set is frozen to the 101 primary weeks used by the main allocation
    analysis.  National forecasts are paired with the same spatial policies; a
    separate direct divisional log-growth baseline bypasses national-to-division
    share disaggregation.
    """
    target_rows = joint[
        joint["analysis_rule"].eq("published_primary")
        & joint["alignment_scenario"].eq("confirmed_dghs")
        & joint["forecast_level"].eq("point")
        & joint["policy"].eq("pooled3")
    ][["selected_year", "published_week_num", "published_week_label", "target_id"]].drop_duplicates()
    target_ids = set(target_rows["target_id"])

    predictions = pd.read_csv(PREDICTION_PATH, parse_dates=["target_week_start"])
    models = list(dict.fromkeys([selected_model, "baseline_hgb", "rolling_selected_all"]))
    predictions = predictions[
        predictions["horizon"].eq(1)
        & predictions["model"].isin(models)
        & predictions["prediction"].notna()
    ].copy()
    fields = predictions["target_week_start"].dt.date.map(dghs_week_fields)
    predictions[["week_year", "week_number"]] = pd.DataFrame(
        fields.tolist(), index=predictions.index
    )
    prediction_lookup = {
        (str(row.model), int(row.week_year), int(row.week_number)): float(row.prediction)
        for row in predictions.itertuples()
    }

    matrix = eligible_matrix(
        load_spatial_matrix(), pd.read_csv(WEEK_AUDIT_PATH), "published_primary"
    )
    matrix_lookup: dict[tuple[int, int], tuple[pd.Series, pd.DataFrame, pd.DataFrame]] = {}
    for year, year_data in matrix.groupby("selected_year", sort=True):
        ordered = year_data.sort_values("week_num").reset_index(drop=True)
        for position in range(len(ordered)):
            target = ordered.iloc[position]
            matrix_lookup[(int(year), int(target["week_num"]))] = (
                target, ordered.iloc[max(0, position - 3):position],
                ordered.iloc[max(0, position - 4):position],
            )

    rows: list[dict] = []
    for target in target_rows.itertuples():
        key = (int(target.selected_year), int(target.published_week_num))
        spatial_target, history3, history4 = matrix_lookup[key]
        actual = spatial_target[DIVISIONS].astype(float)
        for model in models:
            point = prediction_lookup.get((model, *key))
            if point is None or len(history3) < 3:
                continue
            for policy in ("pooled3", "max3_envelope"):
                allocation_weight, normalized_weight = policy_weights(history3, policy)
                result = allocation_result(
                    actual, allocation_weight, normalized_weight, point
                )
                rows.append({
                    "selected_year": key[0],
                    "published_week_num": key[1],
                    "target_id": target.target_id,
                    "forecast_model": model,
                    "allocation_policy": policy,
                    "actual_total": float(actual.sum()),
                    "allocation_total": result["allocation_total"],
                    "shortage_total": result["shortage_total"],
                    "unused_total": result["unused_total"],
                    "shortage_fraction": result["shortage_fraction"],
                    "unused_fraction_of_allocation": result["unused_fraction"],
                    "budget_to_actual_ratio": result["budget_to_actual_ratio"],
                    "worst_division_service_fraction": result["worst_service"],
                    "full_division_coverage": result["full_coverage"],
                })
        if len(history4) == 4:
            direct = pd.Series(
                {
                    division: max(
                        0.0,
                        float(
                            np.expm1(
                                np.log1p(history4[division].astype(float).iloc[-1])
                                + np.polyfit(
                                    np.arange(4, dtype=float),
                                    np.log1p(history4[division].astype(float).to_numpy()),
                                    1,
                                )[0]
                            )
                        ),
                    )
                    for division in DIVISIONS
                }
            )
            result = direct_allocation_result(actual, direct)
            rows.append({
                "selected_year": key[0],
                "published_week_num": key[1],
                "target_id": target.target_id,
                "forecast_model": "direct_division_log_growth4",
                "allocation_policy": "direct",
                "actual_total": float(actual.sum()),
                "allocation_total": result["allocation_total"],
                "shortage_total": result["shortage_total"],
                "unused_total": result["unused_total"],
                "shortage_fraction": result["shortage_fraction"],
                "unused_fraction_of_allocation": result["unused_fraction"],
                "budget_to_actual_ratio": result["budget_to_actual_ratio"],
                "worst_division_service_fraction": result["worst_service"],
                "full_division_coverage": result["full_coverage"],
            })
    detail = pd.DataFrame(rows)

    def summarize_group(group: pd.DataFrame) -> pd.Series:
        actual_total = group["actual_total"].sum()
        allocation_total = group["allocation_total"].sum()
        shortage_total = group["shortage_total"].sum()
        return pd.Series({
            "target_weeks": group["target_id"].nunique(),
            "mean_shortage_fraction": group["shortage_fraction"].mean(),
            "demand_weighted_shortage_fraction": shortage_total / actual_total,
            "total_shortage_admission_equivalents": shortage_total,
            "mean_weekly_shortage_admission_equivalents": group["shortage_total"].mean(),
            "p90_shortage_fraction": group["shortage_fraction"].quantile(0.90),
            "mean_unused_fraction_of_allocation": group[
                "unused_fraction_of_allocation"
            ].mean(),
            "aggregate_budget_to_actual_ratio": allocation_total / actual_total,
            "mean_budget_to_actual_ratio": group["budget_to_actual_ratio"].mean(),
            "mean_worst_division_service_fraction": group[
                "worst_division_service_fraction"
            ].mean(),
            "full_division_coverage_rate": group["full_division_coverage"].mean(),
        })

    summary = detail.groupby(
        ["forecast_model", "allocation_policy"], as_index=False
    ).apply(summarize_group, include_groups=False).reset_index(drop=True)
    expected = len(target_ids)
    national_rows = summary[~summary["forecast_model"].eq("direct_division_log_growth4")]
    if not national_rows["target_weeks"].eq(expected).all():
        raise RuntimeError("National forecast sensitivity did not retain the frozen target set.")
    # Direct four-week division trends lose W04 after each year boundary because
    # the primary share analysis requires only three within-year history weeks.
    # Their smaller denominator is retained and reported rather than backfilled.
    return detail, summary


def write_report(
    summary: pd.DataFrame,
    frontier: pd.DataFrame,
    comparisons: pd.DataFrame,
    selected_model: str,
) -> None:
    primary = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["alignment_scenario"].eq("confirmed_dghs")
    ].set_index(["forecast_level", "policy"])
    oracle_pooled = primary.loc[("observed_total_oracle", "pooled3")]
    point_pooled = primary.loc[("point", "pooled3")]
    point_envelope = primary.loc[("point", "max3_envelope")]
    upper_pooled = primary.loc[("upper80", "pooled3")]
    upper_envelope = primary.loc[("upper80", "max3_envelope")]
    alignment_slice = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["forecast_level"].eq("point")
        & summary["policy"].eq("pooled3")
    ]
    shortage_min = alignment_slice["mean_shortage_fraction"].min()
    shortage_max = alignment_slice["mean_shortage_fraction"].max()
    fixed20 = frontier.set_index("reserve_rule").loc["point_plus_20pct"]
    fixed30 = frontier.set_index("reserve_rule").loc["point_plus_30pct"]

    lines = [
        "# Joint forecast-allocation hindcast",
        "",
        f"Generated: {RUN_DATE}",
        "",
        "## Main result",
        "",
        (
            f"The joint workflow is technically workable as a research prototype. Using the selected one-week "
            f"{selected_model} forecast, confirmed DGHS Sunday-Saturday alignment, and trailing 3-week pooled division shares, "
            f"the {int(point_pooled['target_weeks'])}-week hindcast had {point_pooled['mean_shortage_fraction']:.1%} "
            f"mean divisional shortage and {point_pooled['p90_shortage_fraction']:.1%} P90 shortage. Mean absolute "
            f"shortfall was {point_pooled['mean_weekly_shortage_admission_equivalents']:.1f} admissions per week, and total "
            f"allocation was {point_pooled['aggregate_budget_to_actual_ratio']:.1%} of total division-reported admissions."
        ),
        "",
        (
            f"With the observed national total supplied as a diagnostic oracle, pooled shares alone produced "
            f"{oracle_pooled['mean_shortage_fraction']:.1%} mean shortage. The difference to the joint point-forecast "
            f"result shows that national forecast and aggregation mismatch add material operational error beyond the "
            f"spatial-share problem."
        ),
        "",
        "## Reserve trade-offs",
        "",
        "| Strategy | Mean shortage | P90 shortage | Mean unused allocation | Total budget / total actual | Full division coverage |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Point forecast + pooled shares | {point_pooled['mean_shortage_fraction']:.1%} | "
            f"{point_pooled['p90_shortage_fraction']:.1%} | {point_pooled['mean_unused_fraction_of_allocation']:.1%} | "
            f"{point_pooled['aggregate_budget_to_actual_ratio']:.1%} | {point_pooled['full_division_coverage_rate']:.1%} |"
        ),
        (
            f"| Point forecast + max-share envelope | {point_envelope['mean_shortage_fraction']:.1%} | "
            f"{point_envelope['p90_shortage_fraction']:.1%} | {point_envelope['mean_unused_fraction_of_allocation']:.1%} | "
            f"{point_envelope['aggregate_budget_to_actual_ratio']:.1%} | {point_envelope['full_division_coverage_rate']:.1%} |"
        ),
        (
            f"| Upper 80% bound + pooled shares | {upper_pooled['mean_shortage_fraction']:.1%} | "
            f"{upper_pooled['p90_shortage_fraction']:.1%} | {upper_pooled['mean_unused_fraction_of_allocation']:.1%} | "
            f"{upper_pooled['aggregate_budget_to_actual_ratio']:.1%} | {upper_pooled['full_division_coverage_rate']:.1%} |"
        ),
        (
            f"| Upper 80% bound + max-share envelope | {upper_envelope['mean_shortage_fraction']:.1%} | "
            f"{upper_envelope['p90_shortage_fraction']:.1%} | {upper_envelope['mean_unused_fraction_of_allocation']:.1%} | "
            f"{upper_envelope['aggregate_budget_to_actual_ratio']:.1%} | {upper_envelope['full_division_coverage_rate']:.1%} |"
        ),
        "",
        (
            f"A fixed 20% uplift on the point forecast reduced mean shortage to "
            f"{fixed20['mean_shortage_fraction']:.1%} with an aggregate budget of "
            f"{fixed20['aggregate_budget_to_actual_ratio']:.1%} of actual. A 30% uplift reduced it to "
            f"{fixed30['mean_shortage_fraction']:.1%}, with a mean budget ratio of "
            f"{fixed30['aggregate_budget_to_actual_ratio']:.1%}. These are trade-off points, not recommended optima; "
            f"choosing one requires shortage and holding-cost values."
        ),
        "",
        "## Week alignment",
        "",
        "DGHS Wxx labels and all national/weather aggregates use Sunday-Saturday weeks; week 1 contains 1 January. The confirmed example 9-15 August 2026 = W33 is reproduced by the calendar function.",
        "",
        "## Paired uncertainty comparisons",
        "",
    ]
    for row in comparisons.itertuples():
        lines.append(
            f"- {row.candidate_strategy_label} minus {row.reference_strategy_label}: mean shortage difference "
            f"{row.observed_mean_shortage_difference:+.2%}; moving-block bootstrap 95% interval "
            f"[{row.shortage_difference_95pct_ci_lower:+.2%}, {row.shortage_difference_95pct_ci_upper:+.2%}]. "
            f"Mean budget-ratio difference {row.observed_mean_budget_ratio_difference:+.1%}."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The point forecast with pooled shares is the lowest-complexity operational baseline. The point-plus-envelope strategy is a moderate contingency option. Using the full upper conformal bound every week substantially reduces shortage, but it also produces large unused allocations and is too conservative to call a routine allocation rule without explicit inventory, expiry, transfer, and shortage costs.",
        "",
        "## Limitations",
        "",
        "- The analysis uses only one-week forecast targets with non-missing conformal intervals; the early 2024 calibration period is excluded.",
        "- The Sunday-Saturday Wxx convention was supplied by the study team and verified against the example 9-15 August 2026 = W33; an official machine-readable calendar table was not available.",
        "- The forecast target and spatial actuals are both DGHS hospital-admission aggregates, but their totals can differ because they come from different dashboard series and calendar conventions.",
        "- Admissions omit unknown unmet, diverted, non-hospital, and community demand.",
        "- Admission-equivalent allocations do not prove facility feasibility. Usable dengue beds, IV stocks, lead times, workforce, referrals, transport, and transfer constraints remain absent.",
    ])
    (OUTPUTS / "dengue_phase2_joint_backtest_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_checkpoint(summary: pd.DataFrame, frontier: pd.DataFrame, selected_model: str) -> None:
    primary = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["alignment_scenario"].eq("confirmed_dghs")
    ].set_index(["forecast_level", "policy"])
    point = primary.loc[("point", "pooled3")]
    envelope = primary.loc[("point", "max3_envelope")]
    upper = primary.loc[("upper80", "pooled3")]
    text = f"""# Joint forecast-allocation checkpoint

Last updated: {RUN_DATE} (Asia/Dhaka)

## Completed

- Selected one-week model `{selected_model}` joined to DGHS division-share policies.
- Confirmed DGHS Sunday-Saturday Wxx mapping with week 1 containing 1 January.
- Primary and isolated-low sensitivity datasets; no missing target was imputed.
- Point, upper-80%, and observed-total diagnostic forecast levels.
- Paired year-stratified moving-block bootstrap and a fixed-reserve frontier.
- Exact aggregate-budget matching with intervals for mean, weighted, and surge shortage.
- HGB, prospective rolling-selector, and direct-divisional forecast sensitivities.

## Primary causal-alignment results

- Point forecast + pooled shares: {point['mean_shortage_fraction']:.3%} mean shortage; {point['aggregate_budget_to_actual_ratio']:.3%} total budget/total actual.
- Point forecast + max-share envelope: {envelope['mean_shortage_fraction']:.3%} mean shortage; {envelope['aggregate_budget_to_actual_ratio']:.3%} total budget/total actual.
- Upper-80% + pooled shares: {upper['mean_shortage_fraction']:.3%} mean shortage; {upper['aggregate_budget_to_actual_ratio']:.3%} total budget/total actual.

## Decision status

- Workable research prototype: yes.
- Week-calendar blocker: resolved using the confirmed Sunday-Saturday definition.
- Routine baseline for further work: point forecast + trailing 3-week pooled shares.
- Reserve policy remains a cost trade-off; no optimum is asserted without shortage, holding, expiry, transfer, and lead-time costs.

## Resume

1. `python3 src/backtest_joint_forecast_allocation.py`
2. `node work/phase2_capacity/build_phase2_workbook.mjs`
3. Next data collection: facility-week usable dengue beds, IV inventory/lead time, staff rosters or WISN inputs, transfers, and denied/diverted admissions.
"""
    (WORK / "JOINT_BACKTEST_CHECKPOINT.md").write_text(text, encoding="utf-8")


def main() -> None:
    joint, division_detail, alignment_decisions, week_audit, selected_model = run_joint_backtest()
    summary = summarize(joint)
    reserve_rows = reserve_frontier(joint)
    frontier, frontier_detail = complete_reserve_frontier(
        reserve_rows, division_detail
    )
    matched_comparisons = budget_matched_comparisons(frontier_detail)
    comparisons = paired_bootstrap(joint)
    forecast_sensitivity_detail, forecast_sensitivity = allocation_forecast_sensitivity(
        joint, selected_model
    )

    output_frames = {
        "outputs/dengue_phase2_joint_backtest_detail.csv": joint,
        "outputs/dengue_phase2_joint_backtest_division_detail.csv": division_detail,
        "outputs/dengue_phase2_joint_backtest_summary.csv": summary,
        "outputs/dengue_phase2_joint_bootstrap_comparisons.csv": comparisons,
        "outputs/dengue_phase2_joint_alignment_decisions.csv": alignment_decisions,
        "outputs/dengue_phase2_joint_reserve_frontier.csv": frontier,
        "outputs/dengue_phase2_joint_reserve_frontier_detail.csv": frontier_detail,
        "outputs/dengue_phase2_joint_budget_matched_comparisons.csv": matched_comparisons,
        "outputs/dengue_phase2_forecast_choice_sensitivity_detail.csv": forecast_sensitivity_detail,
        "outputs/dengue_phase2_forecast_choice_sensitivity.csv": forecast_sensitivity,
    }
    for relative_path, frame in output_frames.items():
        frame.to_csv(ROOT / relative_path, index=False)
    write_report(summary, frontier, comparisons, selected_model)
    write_checkpoint(summary, frontier, selected_model)

    manifest = {
        "status": "complete",
        "generated_on": RUN_DATE,
        "selected_forecast_model": selected_model,
        "objective": "Jointly evaluate one-week national forecast and divisional allocation policies on the confirmed DGHS Sunday-Saturday calendar.",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                DIVISION_PATH, NATIONAL_PATH, PREDICTION_PATH, SELECTED_MODEL_PATH,
                WEEK_AUDIT_PATH,
            ]
        },
        "parameters": {
            "forecast_horizon_weeks": 1,
            "require_nonmissing_conformal_interval": True,
            "week_definition": "Sunday-Saturday; week 1 contains 1 January",
            "reserve_uplifts": RESERVE_UPLIFTS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_block_length_weeks": BOOTSTRAP_BLOCK_LENGTH,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "outputs": [
            *output_frames.keys(),
            "outputs/dengue_phase2_joint_backtest_report.md",
            "work/phase2_capacity/JOINT_BACKTEST_CHECKPOINT.md",
        ],
        "limitations": [
            "No official machine-readable DGHS week-calendar table was available; the confirmed study definition is encoded and tested.",
            "Division and daily national dashboard aggregates can disagree.",
            "Reported admissions omit unknown unmet and non-hospital demand.",
            "No facility-week stock, usable-bed, staff, transfer, referral, or lead-time constraints are modeled.",
        ],
    }
    (WORK / "joint_backtest_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    primary = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["alignment_scenario"].eq("confirmed_dghs")
        & summary["policy"].isin(["pooled3", "max3_envelope"])
    ]
    print(primary[[
        "forecast_level", "policy", "target_weeks", "mean_shortage_fraction",
        "mean_unused_fraction_of_allocation", "mean_budget_to_actual_ratio",
    ]].to_string(index=False))
    print(f"\nJoint detail rows: {len(joint):,}; division detail rows: {len(division_detail):,}")


if __name__ == "__main__":
    main()
