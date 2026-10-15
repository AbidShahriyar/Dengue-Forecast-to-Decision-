#!/usr/bin/env python3
"""Phase 3: capacity-censored dengue demand and robust allocation stress test.

This analysis does not estimate unobserved patients. It treats reported dengue
admissions as a lower bound and asks whether allocation conclusions survive a
transparent range of hypothetical, spatially heterogeneous censoring patterns.
The only operational capacity snapshot is transcribed from the cited 2023
Transparency International Bangladesh field report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase3_latent_demand"
OUTPUTS = ROOT / "outputs"
DIVISION_DETAIL = OUTPUTS / "dengue_phase2_joint_backtest_division_detail.csv"
CAPACITY_CONTEXT = OUTPUTS / "dengue_capacity_division_context_2025.csv"
RUN_DATE = "2026-10-14"
DIVISIONS = [
    "Barisal", "Chattogram", "Dhaka", "Khulna",
    "Mymensingh", "Rajshahi", "Rangpur", "Sylhet",
]
DELTA_LEVELS = [0.0, 0.25, 0.50, 1.00]
BUDGET_UPLIFTS = [0.0, 0.25, 0.50, 0.75, 1.00]

TIB_URL = (
    "https://ti-bangladesh.org/images/2023/report/dengue-crisis/"
    "Executive-Study-EN-Dengue-Crisis-Prevention-and-Control-TIB.pdf"
)
WHO_SITREP_URL = (
    "https://cdn.who.int/media/docs/default-source/searo/bangladesh/"
    "dengue-sitrep/dengue-sitrep-issue-1.pdf"
)
TIB_LOCAL = WORK / "raw/tib_dengue_crisis_2023.pdf"
WHO_SITREP_LOCAL = WORK / "raw/who_sitrep_1.pdf"
WHO_SENTINEL_PANEL = OUTPUTS / "dengue_who_2023_sentinel_hospital_weekly.csv"
WHO_SENTINEL_AUDIT = OUTPUTS / "dengue_who_2023_sentinel_hospital_audit.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_capacity_snapshot() -> pd.DataFrame:
    """Transcribe the September 2023 TIB field table without filling blanks."""
    rows = [
        ("Dhaka", "Dhaka", "not stated", 2500.0, "reported", 109.0, "reported", 821.0),
        ("Cumilla", "Chattogram", "July 2023", np.nan, "no separate seats", np.nan, "no separate ICU seats", 48.0),
        ("Barisal", "Barisal", "March 2023", np.nan, "no separate seats", np.nan, "no separate ICU seats", 119.0),
        ("Chattogram", "Chattogram", "June 2023", 450.0, "reported", 33.0, "reported", 127.0),
        ("Manikganj", "Dhaka", "May 2023", 210.0, "reported", np.nan, "no separate ICU seats", 118.0),
        ("Faridpur", "Dhaka", "June 2023", 150.0, "reported", 10.0, "reported", 46.0),
        ("Lakshmipur", "Chattogram", "June 2023", 60.0, "reported", 0.0, "reported zero", 68.0),
        ("Chandpur", "Chattogram", "June 2023", 100.0, "reported", 0.0, "reported zero", 46.0),
        ("Pirojpur", "Barisal", "June 2023", np.nan, "no separate seats", 0.0, "reported zero", 59.0),
        ("Patuakhali", "Barisal", "not provided", np.nan, "information not provided", np.nan, "information not provided", np.nan),
    ]
    columns = [
        "district", "division", "preparation_start", "allocated_dengue_seats",
        "seat_source_status", "icu_seats", "icu_source_status", "reported_patients_per_day",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["source_period"] = "September 2023 field snapshot"
    frame["source_page_pdf"] = 19
    frame["source_page_printed"] = 19
    frame["source_url"] = TIB_URL
    frame["los_lower_days"] = 4.0
    frame["los_upper_days"] = 5.0
    frame["implied_concurrent_patients_lower"] = (
        frame["reported_patients_per_day"] * frame["los_lower_days"]
    )
    frame["implied_concurrent_patients_upper"] = (
        frame["reported_patients_per_day"] * frame["los_upper_days"]
    )
    has_seats = frame["allocated_dengue_seats"].notna()
    frame["implied_seat_load_ratio_lower"] = np.where(
        has_seats,
        frame["implied_concurrent_patients_lower"] / frame["allocated_dengue_seats"],
        np.nan,
    )
    frame["implied_seat_load_ratio_upper"] = np.where(
        has_seats,
        frame["implied_concurrent_patients_upper"] / frame["allocated_dengue_seats"],
        np.nan,
    )
    frame["implied_shortfall_lower"] = np.where(
        has_seats,
        (frame["implied_concurrent_patients_lower"] - frame["allocated_dengue_seats"]).clip(lower=0),
        np.nan,
    )
    frame["implied_shortfall_upper"] = np.where(
        has_seats,
        (frame["implied_concurrent_patients_upper"] - frame["allocated_dengue_seats"]).clip(lower=0),
        np.nan,
    )
    frame["interpretation"] = (
        "Illustrative seat-pressure calculation only. 'Patients per day' is retained from the source label; "
        "the 4-5 day relationship comes from the adjacent TIB narrative. These are allocated dengue seats, "
        "not verified staffed beds, and the table is not a complete national facility census."
    )
    return frame


def load_backtest_panel() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series
]:
    detail = pd.read_csv(DIVISION_DETAIL)
    detail = detail[
        detail["analysis_rule"].eq("published_primary")
        & detail["alignment_scenario"].eq("confirmed_dghs")
        & detail["forecast_level"].eq("point")
    ].copy()

    pooled = detail[detail["policy"].eq("pooled3")].copy()
    maxnorm = detail[detail["policy"].eq("max3_normalized")].copy()
    if pooled.duplicated(["target_id", "division"]).any():
        raise RuntimeError("Pooled panel has duplicate target-division rows.")
    if maxnorm.duplicated(["target_id", "division"]).any():
        raise RuntimeError("Max-share panel has duplicate target-division rows.")

    actual = pooled.pivot(index="target_id", columns="division", values="actual_admissions")
    pooled_weights = pooled.pivot(
        index="target_id", columns="division", values="normalized_policy_weight"
    )
    max_weights = maxnorm.pivot(
        index="target_id", columns="division", values="normalized_policy_weight"
    )
    actual = actual[DIVISIONS]
    pooled_weights = pooled_weights[DIVISIONS]
    max_weights = max_weights[DIVISIONS]

    metadata = pooled.groupby("target_id", as_index=True).agg(
        selected_year=("selected_year", "first"),
        published_week_label=("published_week_label", "first"),
        mapped_week_number=("mapped_week_number", "first"),
        mapping_shift=("mapping_shift", "first"),
        history_weeks=("history_weeks", "first"),
        national_point_forecast=("allocated_admission_equivalents", "sum"),
    )
    metadata = metadata.loc[actual.index]

    capacity = pd.read_csv(CAPACITY_CONTEXT).set_index("division").loc[DIVISIONS]
    bor = capacity["bed_weighted_annual_bor_pct"].astype(float)
    stress = (bor - bor.min()) / (bor.max() - bor.min())
    stress.name = "capacity_stress_proxy_0_1"
    return metadata, actual, pooled_weights, max_weights, stress


def stress_profile(stress: pd.Series, profile: str) -> pd.Series:
    if profile == "uniform":
        return pd.Series(1.0, index=DIVISIONS)
    if profile == "capacity_linked":
        return 0.5 + 0.5 * stress
    if profile == "inverse_capacity":
        return 1.0 - 0.5 * stress
    if profile.startswith("rotation_"):
        shift = int(profile.split("_")[1])
        rotated = np.roll(stress.to_numpy(), shift)
        return pd.Series(0.5 + 0.5 * rotated, index=DIVISIONS)
    raise ValueError(profile)


def allocation_metrics(demand: pd.Series, allocation: pd.Series) -> dict[str, float | bool]:
    shortage = (demand - allocation).clip(lower=0)
    unused = (allocation - demand).clip(lower=0)
    service = pd.Series(
        np.where(demand > 0, np.minimum(allocation / demand, 1.0), 1.0),
        index=demand.index,
    )
    total_demand = float(demand.sum())
    total_allocation = float(allocation.sum())
    return {
        "latent_demand": total_demand,
        "allocation_budget": total_allocation,
        "budget_to_latent_demand_ratio": total_allocation / total_demand if total_demand else np.nan,
        "shortage": float(shortage.sum()),
        "shortage_fraction": float(shortage.sum() / total_demand) if total_demand else np.nan,
        "unused": float(unused.sum()),
        "unused_fraction": float(unused.sum() / total_allocation) if total_allocation else np.nan,
        "worst_division_service_fraction": float(service[demand > 0].min()) if (demand > 0).any() else 1.0,
        "full_division_coverage": bool((shortage <= 1e-9).all()),
    }


def build_stress_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata, actual, pooled_weights, max_weights, stress = load_backtest_panel()
    profile_names = [
        "uniform", "capacity_linked", "inverse_capacity",
        *[f"rotation_{shift}" for shift in range(1, 8)],
    ]
    week_rows: list[dict] = []
    division_rows: list[dict] = []

    for target_id in actual.index:
        observed = actual.loc[target_id].astype(float)
        p = pooled_weights.loc[target_id].astype(float)
        p = p / p.sum()
        q = max_weights.loc[target_id].astype(float)
        q = q / q.sum()
        forecast = float(metadata.loc[target_id, "national_point_forecast"])

        for delta in DELTA_LEVELS:
            robust_upper = 1.0 + delta * (0.625 + 0.375 * stress)
            robust_weights = p * robust_upper
            robust_weights = robust_weights / robust_weights.sum()
            half_robust_weights = 0.5 * p + 0.5 * robust_weights
            half_robust_weights = half_robust_weights / half_robust_weights.sum()
            policies = {
                "pooled_reported": p,
                "max_share_normalized": q,
                "capacity_prior_half_weighted": half_robust_weights,
                "capacity_prior_full_weighted": robust_weights,
            }

            for profile in profile_names:
                profile_score = stress_profile(stress, profile)
                multiplier = 1.0 + delta * profile_score
                latent = observed * multiplier
                for budget_uplift in BUDGET_UPLIFTS:
                    budget = forecast * (1.0 + budget_uplift)
                    for policy, weights in policies.items():
                        allocation = budget * weights
                        metrics = allocation_metrics(latent, allocation)
                        row = {
                            "target_id": target_id,
                            **metadata.loc[target_id].to_dict(),
                            "latent_uplift_max": delta,
                            "truth_profile": profile,
                            "budget_uplift_over_point_forecast": budget_uplift,
                            "allocation_policy": policy,
                            **metrics,
                        }
                        week_rows.append(row)
                        for division in DIVISIONS:
                            division_rows.append({
                                "target_id": target_id,
                                "selected_year": int(metadata.loc[target_id, "selected_year"]),
                                "published_week_label": metadata.loc[target_id, "published_week_label"],
                                "latent_uplift_max": delta,
                                "truth_profile": profile,
                                "budget_uplift_over_point_forecast": budget_uplift,
                                "allocation_policy": policy,
                                "division": division,
                                "observed_admissions_lower_bound": float(observed[division]),
                                "latent_multiplier_scenario": float(multiplier[division]),
                                "latent_demand_scenario": float(latent[division]),
                                "allocation_weight": float(weights[division]),
                                "allocated_admission_equivalents": float(allocation[division]),
                                "capacity_stress_proxy_0_1": float(stress[division]),
                            })

    week = pd.DataFrame(week_rows)
    division = pd.DataFrame(division_rows)
    group_columns = [
        "latent_uplift_max", "truth_profile",
        "budget_uplift_over_point_forecast", "allocation_policy",
    ]
    summary = (
        week.groupby(group_columns, as_index=False)
        .agg(
            comparable_weeks=("target_id", "nunique"),
            mean_shortage_fraction=("shortage_fraction", "mean"),
            p90_shortage_fraction=("shortage_fraction", lambda values: values.quantile(0.90)),
            mean_unused_fraction=("unused_fraction", "mean"),
            mean_worst_division_service_fraction=("worst_division_service_fraction", "mean"),
            full_division_coverage_fraction=("full_division_coverage", "mean"),
            mean_budget_to_latent_demand_ratio=("budget_to_latent_demand_ratio", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return week, division, summary


def build_policy_comparisons(week: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "target_id", "latent_uplift_max", "truth_profile",
        "budget_uplift_over_point_forecast",
    ]
    pivot = week.pivot_table(
        index=index_columns,
        columns="allocation_policy",
        values=["shortage_fraction", "worst_division_service_fraction"],
        aggfunc="first",
    )
    rows: list[dict] = []
    for delta in DELTA_LEVELS:
        for budget_uplift in BUDGET_UPLIFTS:
            subset = pivot.loc[
                (slice(None), delta, slice(None), budget_uplift), :
            ].reset_index()
            for challenger in [
                "max_share_normalized",
                "capacity_prior_half_weighted",
                "capacity_prior_full_weighted",
            ]:
                shortage_diff = (
                    subset[("shortage_fraction", challenger)]
                    - subset[("shortage_fraction", "pooled_reported")]
                )
                service_diff = (
                    subset[("worst_division_service_fraction", challenger)]
                    - subset[("worst_division_service_fraction", "pooled_reported")]
                )
                rows.append({
                    "latent_uplift_max": delta,
                    "budget_uplift_over_point_forecast": budget_uplift,
                    "challenger_policy": challenger,
                    "scenario_week_count": int(len(subset)),
                    "mean_shortage_difference_pp_vs_pooled": float(shortage_diff.mean() * 100),
                    "worst_scenario_week_shortage_difference_pp_vs_pooled": float(shortage_diff.max() * 100),
                    "best_scenario_week_shortage_difference_pp_vs_pooled": float(shortage_diff.min() * 100),
                    "fraction_scenario_weeks_lower_shortage_than_pooled": float((shortage_diff < -1e-12).mean()),
                    "mean_worst_service_difference_pp_vs_pooled": float(service_diff.mean() * 100),
                    "fraction_scenario_weeks_higher_worst_service_than_pooled": float((service_diff > 1e-12).mean()),
                })
    return pd.DataFrame(rows)


def build_minimax_proxy_audit() -> pd.DataFrame:
    """Test whether any weight on the capacity proxy is ambiguity-minimax.

    The candidate policy is a convex mixture of pooled reported-demand shares
    and the capacity-prior full-weighted shares. The loss is the worst L1
    distance to forecast-demand shares over uniform, capacity-linked, inverse,
    and seven rotated spatial stress patterns. This is a policy-design audit, not an empirical
    estimate of the latent spatial pattern.
    """
    _, _, pooled_weights, _, stress = load_backtest_panel()
    profiles = [
        "uniform", "capacity_linked", "inverse_capacity",
        *[f"rotation_{shift}" for shift in range(1, 8)],
    ]
    rho_grid = np.linspace(0.0, 1.0, 21)
    rows: list[dict] = []
    for target_id, raw_p in pooled_weights.iterrows():
        p = raw_p.astype(float)
        p = p / p.sum()
        for delta in DELTA_LEVELS:
            capacity_factor = 1.0 + delta * (0.625 + 0.375 * stress)
            capacity_weighted = p * capacity_factor
            capacity_weighted = capacity_weighted / capacity_weighted.sum()
            scenario_weights = []
            for profile in profiles:
                factor = 1.0 + delta * stress_profile(stress, profile)
                scenario = p * factor
                scenario_weights.append((scenario / scenario.sum()).to_numpy())
            candidate_rows = []
            for rho in rho_grid:
                candidate = (1.0 - rho) * p + rho * capacity_weighted
                distances = [
                    float(np.abs(candidate.to_numpy() - scenario).sum())
                    for scenario in scenario_weights
                ]
                candidate_rows.append({
                    "target_id": target_id,
                    "latent_uplift_max": delta,
                    "capacity_proxy_weight": float(rho),
                    "worst_case_l1_share_distance": max(distances),
                    "mean_l1_share_distance": float(np.mean(distances)),
                })
            if delta == 0.0:
                # Every mixture is theoretically identical when no latent
                # uplift exists. Select zero explicitly instead of allowing
                # floating-point noise to choose a nonzero proxy weight.
                best = candidate_rows[0]
            else:
                best = min(
                    candidate_rows,
                    key=lambda row: (
                        round(row["worst_case_l1_share_distance"], 14),
                        round(row["mean_l1_share_distance"], 14),
                        row["capacity_proxy_weight"],
                    ),
                )
            for row in candidate_rows:
                row["selected_minimax"] = (
                    row["capacity_proxy_weight"] == best["capacity_proxy_weight"]
                )
                rows.append(row)
    return pd.DataFrame(rows)


def build_decision_aligned_minimax_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate capacity-proxy mixtures using shortage and regret objectives.

    Unlike the share-distance diagnostic, this audit scores the downstream
    allocation loss directly. A single proxy weight is evaluated across all
    historical target weeks for each latent-uplift and reserve setting. The
    ambiguity set contains the same ten declared spatial stress patterns.
    """
    metadata, actual, pooled_weights, _, stress = load_backtest_panel()
    profiles = [
        "uniform", "capacity_linked", "inverse_capacity",
        *[f"rotation_{shift}" for shift in range(1, 8)],
    ]
    rho_grid = np.linspace(0.0, 1.0, 21)
    rows: list[dict] = []

    for delta in DELTA_LEVELS:
        capacity_factor = 1.0 + delta * (0.625 + 0.375 * stress)
        for budget_uplift in BUDGET_UPLIFTS:
            scenario_results: dict[float, dict[str, tuple[float, float]]] = {}
            for rho in rho_grid:
                by_profile: dict[str, tuple[float, float]] = {}
                for profile in profiles:
                    week_shortages: list[float] = []
                    shortage_totals: list[float] = []
                    demand_totals: list[float] = []
                    profile_score = stress_profile(stress, profile)
                    for target_id in actual.index:
                        observed = actual.loc[target_id].astype(float)
                        latent = observed * (1.0 + delta * profile_score)
                        p = pooled_weights.loc[target_id].astype(float)
                        p = p / p.sum()
                        capacity_weighted = p * capacity_factor
                        capacity_weighted = capacity_weighted / capacity_weighted.sum()
                        candidate = (1.0 - rho) * p + rho * capacity_weighted
                        budget = float(
                            metadata.loc[target_id, "national_point_forecast"]
                        ) * (1.0 + budget_uplift)
                        allocation = budget * candidate
                        shortage = (latent - allocation).clip(lower=0)
                        week_shortages.append(float(shortage.sum() / latent.sum()))
                        shortage_totals.append(float(shortage.sum()))
                        demand_totals.append(float(latent.sum()))
                    by_profile[profile] = (
                        float(np.mean(week_shortages)),
                        float(np.sum(shortage_totals) / np.sum(demand_totals)),
                    )
                scenario_results[float(rho)] = by_profile

            best_by_profile = {
                profile: min(
                    scenario_results[float(rho)][profile][0] for rho in rho_grid
                )
                for profile in profiles
            }
            block_rows: list[dict] = []
            for rho in rho_grid:
                by_profile = scenario_results[float(rho)]
                mean_shortages = [by_profile[profile][0] for profile in profiles]
                weighted_shortages = [by_profile[profile][1] for profile in profiles]
                regrets = [
                    by_profile[profile][0] - best_by_profile[profile]
                    for profile in profiles
                ]
                block_rows.append({
                    "latent_uplift_max": delta,
                    "budget_uplift_over_point_forecast": budget_uplift,
                    "capacity_proxy_weight": float(rho),
                    "target_weeks": int(len(actual)),
                    "spatial_stress_profiles": int(len(profiles)),
                    "worst_profile_mean_shortage_fraction": float(max(mean_shortages)),
                    "mean_profile_mean_shortage_fraction": float(np.mean(mean_shortages)),
                    "worst_profile_demand_weighted_shortage_fraction": float(
                        max(weighted_shortages)
                    ),
                    "worst_profile_shortage_regret_fraction": float(max(regrets)),
                    "mean_profile_shortage_regret_fraction": float(np.mean(regrets)),
                })
            best_shortage = min(
                block_rows,
                key=lambda row: (
                    round(row["worst_profile_mean_shortage_fraction"], 14),
                    round(row["mean_profile_mean_shortage_fraction"], 14),
                    row["capacity_proxy_weight"],
                ),
            )
            best_regret = min(
                block_rows,
                key=lambda row: (
                    round(row["worst_profile_shortage_regret_fraction"], 14),
                    round(row["mean_profile_shortage_regret_fraction"], 14),
                    row["capacity_proxy_weight"],
                ),
            )
            for row in block_rows:
                row["selected_minimax_shortage"] = (
                    row["capacity_proxy_weight"]
                    == best_shortage["capacity_proxy_weight"]
                )
                row["selected_minimax_regret"] = (
                    row["capacity_proxy_weight"]
                    == best_regret["capacity_proxy_weight"]
                )
                rows.append(row)

    audit = pd.DataFrame(rows)
    summary_rows: list[dict] = []
    for criterion, flag in [
        ("worst_case_shortage", "selected_minimax_shortage"),
        ("worst_case_regret", "selected_minimax_regret"),
    ]:
        selected = audit[audit[flag]].copy()
        for row in selected.itertuples():
            summary_rows.append({
                "criterion": criterion,
                "latent_uplift_max": row.latent_uplift_max,
                "budget_uplift_over_point_forecast": row.budget_uplift_over_point_forecast,
                "selected_capacity_proxy_weight": row.capacity_proxy_weight,
                "worst_profile_mean_shortage_fraction": row.worst_profile_mean_shortage_fraction,
                "mean_profile_mean_shortage_fraction": row.mean_profile_mean_shortage_fraction,
                "worst_profile_demand_weighted_shortage_fraction": row.worst_profile_demand_weighted_shortage_fraction,
                "worst_profile_shortage_regret_fraction": row.worst_profile_shortage_regret_fraction,
            })
    return audit, pd.DataFrame(summary_rows)


def build_data_request() -> pd.DataFrame:
    rows = [
        (1, "Demand censoring", "All dengue-suspected/confirmed emergency or outpatient presentations; admitted; refused/no-bed; diverted; waitlisted", "facility-day", "Critical", "Identifies the admitted fraction and directly tests whether observed admissions are capacity-censored", "Without it, latent demand remains an assumption-bounded sensitivity, not an estimated outcome"),
        (2, "Usable capacity", "Staffed dengue ward and ICU beds, occupied beds, temporary/floor beds, closures", "facility-day or shift", "Critical", "Converts predicted admission need into feasible bed activation and overflow", "Approved beds and annual occupancy cannot support a real weekly bed decision"),
        (3, "Patient flow", "Arrival, admission, discharge, transfer and death timestamps; ward/ICU; severity or warning-sign class", "deidentified encounter", "Critical", "Estimates length-of-stay distributions, occupancy, queueing and transfer needs", "Beds can only be reported as scenario equivalents, not dynamic occupancy"),
        (4, "Referral network", "Origin facility, destination facility, request time, acceptance/rejection, transport time", "deidentified transfer", "High", "Supports network allocation and validates spatial overflow", "Division-level allocation cannot represent feasible patient transfers"),
        (5, "IV logistics", "Opening stock, receipts, issues/administration, wastage, stockouts, order and delivery dates by fluid type and bag size", "facility-day", "Critical for IV claim", "Estimates consumption and replenishment lead-time distributions", "Litres or bags remain guideline-based scenarios and cannot be optimized as inventory"),
        (6, "Clinical workload", "Nurse-hours and doctor-hours on duty; dengue patient contacts; shift rosters; absences", "facility-shift", "Critical for staffing claim", "Converts census and acuity into measured staffing constraints", "Doctors and nurses cannot be allocated credibly from case counts alone"),
        (7, "Decision costs", "Bed activation, transfer, holding, shortage, overtime and emergency procurement costs", "facility-week or event", "High", "Enables cost-optimal service levels and a defensible objective function", "Any reserve level is a trade-off frontier, not a proven optimum"),
        (8, "Calendar documentation", "Citable official definition of DGHS dashboard Wxx labels and reporting cut-off", "metadata", "Desirable", "Documents the confirmed Sunday-Saturday mapping in an external source", "The study-team definition is encoded and tested, but a citable official calendar note remains desirable"),
        (9, "Facility identity", "Stable facility code, district/division, coordinates, ownership and level", "facility", "High", "Joins demand, capacity and travel-time data reproducibly", "Public hospital records cannot be linked reliably across sources"),
    ]
    columns = [
        "priority", "data_domain", "minimum_requested_fields", "minimum_granularity",
        "importance", "major_claim_enabled", "consequence_if_missing",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["privacy_minimum"] = (
        "No names, phone numbers, national IDs or exact home addresses are required; stable pseudonymous encounter IDs are sufficient where linkage is needed."
    )
    return frame


def build_report(
    snapshot: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    minimax: pd.DataFrame,
) -> str:
    numeric = snapshot[snapshot["allocated_dengue_seats"].notna()].copy()
    seats = numeric["allocated_dengue_seats"].sum()
    patients = numeric["reported_patients_per_day"].sum()
    lower = numeric["implied_concurrent_patients_lower"].sum()
    upper = numeric["implied_concurrent_patients_upper"].sum()
    short_low = max(lower - seats, 0)
    short_high = max(upper - seats, 0)

    key = summary[
        summary["latent_uplift_max"].eq(0.50)
        & summary["truth_profile"].eq("capacity_linked")
        & summary["budget_uplift_over_point_forecast"].eq(0.50)
    ].set_index("allocation_policy")
    comp = comparisons[
        comparisons["latent_uplift_max"].eq(0.50)
        & comparisons["budget_uplift_over_point_forecast"].eq(0.50)
    ].set_index("challenger_policy")

    pooled_short = key.loc["pooled_reported", "mean_shortage_fraction"] * 100
    robust_short = key.loc["capacity_prior_full_weighted", "mean_shortage_fraction"] * 100
    pooled_service = key.loc["pooled_reported", "mean_worst_division_service_fraction"] * 100
    robust_service = key.loc["capacity_prior_full_weighted", "mean_worst_division_service_fraction"] * 100
    robust_mean_diff = comp.loc[
        "capacity_prior_full_weighted", "mean_shortage_difference_pp_vs_pooled"
    ]
    robust_worst_diff = comp.loc[
        "capacity_prior_full_weighted", "worst_scenario_week_shortage_difference_pp_vs_pooled"
    ]
    minimax_selected = minimax[minimax["selected_minimax"]]
    zero_proxy_fraction = float(
        minimax_selected["capacity_proxy_weight"].eq(0.0).mean()
    )
    target_weeks = int(summary["comparable_weeks"].max())
    if WHO_SENTINEL_PANEL.exists():
        sentinel = pd.read_csv(WHO_SENTINEL_PANEL)
        sentinel_description = (
            f"The official reports also yield a reproducible sentinel panel of "
            f"{len(sentinel):,} hospital-week rows across {sentinel['hospital'].nunique()} hospitals "
            f"and {sentinel['epi_week'].nunique()} consecutive peak-season weeks (weeks "
            f"{sentinel['epi_week'].min()}-{sentinel['epi_week'].max()}). Issue 7 publishes only "
            "five hospitals per group; the other issues publish ten. All 100 comparable "
            "consecutive cumulative-case differences reconcile exactly to the reported weekly "
            "cases. The panel is suitable for descriptive triangulation, not independent model "
            "training or validation, and contains no staffed-bed, occupancy, refusal, transfer, "
            "stock, or staffing fields."
        )
    else:
        sentinel_description = (
            "Later WHO reports contain a short sentinel-hospital series, useful for "
            "triangulation but too short for a standalone facility forecasting study."
        )

    return f"""# Phase 3 publication viability and capacity-censoring stress test

Run date: {RUN_DATE}

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
reports {patients:,.0f} patients per day and {seats:,.0f} allocated dengue seats.
Using the report's adjacent 4-5-day stay relationship implies {lower:,.0f} to
{upper:,.0f} concurrent patients, or an illustrative shortfall of {short_low:,.0f}
to {short_high:,.0f} places. This is evidence of severe pressure, not a national
estimate: the source label 'patients per day' is ambiguous, allocated seats are
not verified staffed beds, and the selected districts are not a census.

WHO's first 2023 situation report independently states that COVID-19 beds were
repurposed for dengue and that reported admissions placed additional pressure on
the health system. {sentinel_description}

## Stress-test design

For each of {target_weeks} one-week hindcast targets, observed divisional admissions are
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
shares have {pooled_short:.2f}% mean shortage and {pooled_service:.2f}% mean
worst-division service. The full capacity-prior rule has {robust_short:.2f}% mean
shortage and {robust_service:.2f}% mean worst-division service. The apparent
shortage gain is only {(pooled_short - robust_short):.3f} percentage points.

That rule fails the misspecification test. Across every spatial stress pattern in
the same block, its average shortage difference from pooled allocation is
{robust_mean_diff:+.3f} percentage points and its worst scenario-week difference
is {robust_worst_diff:+.3f} percentage points. A minimax mixture audit selected
zero weight on the capacity proxy in {zero_proxy_fraction:.1%} of target/uplift
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
  2023, pp. 18-20. {TIB_URL}
- WHO Bangladesh. *Dengue Situation Report, Issue 1*, 28 August 2023.
  {WHO_SITREP_URL}
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
"""


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    snapshot = build_capacity_snapshot()
    week, division, summary = build_stress_test()
    comparisons = build_policy_comparisons(week)
    minimax = build_minimax_proxy_audit()
    decision_minimax, decision_minimax_summary = build_decision_aligned_minimax_audit()
    data_request = build_data_request()

    output_paths = {
        "capacity_snapshot": OUTPUTS / "dengue_phase3_capacity_stress_snapshot.csv",
        "week_detail": OUTPUTS / "dengue_phase3_censoring_stress_week_detail.csv",
        "division_detail": OUTPUTS / "dengue_phase3_censoring_stress_division_detail.csv",
        "summary": OUTPUTS / "dengue_phase3_censoring_stress_summary.csv",
        "comparisons": OUTPUTS / "dengue_phase3_censoring_policy_comparisons.csv",
        "minimax_proxy_audit": OUTPUTS / "dengue_phase3_minimax_proxy_weight.csv",
        "decision_minimax_audit": OUTPUTS / "dengue_phase3_decision_minimax_proxy_weight.csv",
        "decision_minimax_summary": OUTPUTS / "dengue_phase3_decision_minimax_summary.csv",
        "data_request": OUTPUTS / "dengue_phase3_minimum_data_request.csv",
        "report": OUTPUTS / "dengue_phase3_publication_viability_report.md",
    }
    snapshot.to_csv(output_paths["capacity_snapshot"], index=False)
    week.to_csv(output_paths["week_detail"], index=False)
    division.to_csv(output_paths["division_detail"], index=False)
    summary.to_csv(output_paths["summary"], index=False)
    comparisons.to_csv(output_paths["comparisons"], index=False)
    minimax.to_csv(output_paths["minimax_proxy_audit"], index=False)
    decision_minimax.to_csv(output_paths["decision_minimax_audit"], index=False)
    decision_minimax_summary.to_csv(
        output_paths["decision_minimax_summary"], index=False
    )
    data_request.to_csv(output_paths["data_request"], index=False)
    output_paths["report"].write_text(
        build_report(snapshot, summary, comparisons, minimax), encoding="utf-8"
    )

    target_weeks = int(summary["comparable_weeks"].max())
    checkpoint = f"""# Phase 3 latent-demand checkpoint

Last updated: {RUN_DATE}

## Completed

- Transcribed the September 2023 TIB dengue seat/ICU/patient snapshot without
  converting unavailable values to zero.
- Calculated transparent 4-5-day concurrent-patient stress ranges for locations
  with numeric allocated dengue seats.
- Ran a deterministic {target_weeks}-week partial-identification stress test over four
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
"""
    checkpoint_path = WORK / "CHECKPOINT.md"
    checkpoint_path.write_text(checkpoint, encoding="utf-8")

    manifest = {
        "run_date": RUN_DATE,
        "method": "capacity-censored partial-identification allocation stress test",
        "deterministic": True,
        "inputs": {
            str(DIVISION_DETAIL.relative_to(ROOT)): sha256(DIVISION_DETAIL),
            str(CAPACITY_CONTEXT.relative_to(ROOT)): sha256(CAPACITY_CONTEXT),
            "tib_source_url": TIB_URL,
            "who_source_url": WHO_SITREP_URL,
        },
        "parameters": {
            "latent_uplift_max": DELTA_LEVELS,
            "budget_uplift_over_point_forecast": BUDGET_UPLIFTS,
            "spatial_truth_profiles": [
                "uniform", "capacity_linked", "inverse_capacity", "rotation_1_to_7"
            ],
            "allocation_policies": [
                "pooled_reported", "max_share_normalized",
                "capacity_prior_half_weighted", "capacity_prior_full_weighted",
            ],
        },
        "outputs": {},
    }
    for source_path in [TIB_LOCAL, WHO_SITREP_LOCAL]:
        if source_path.exists():
            manifest["inputs"][str(source_path.relative_to(ROOT))] = sha256(source_path)
    for source_path in [WHO_SENTINEL_PANEL, WHO_SENTINEL_AUDIT]:
        if source_path.exists():
            manifest["inputs"][str(source_path.relative_to(ROOT))] = sha256(source_path)
    for name, path in output_paths.items():
        manifest["outputs"][name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
    manifest["outputs"]["checkpoint"] = {
        "path": str(checkpoint_path.relative_to(ROOT)),
        "sha256": sha256(checkpoint_path),
    }
    (WORK / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(output_paths)} Phase 3 outputs.")
    print(f"Week rows: {len(week):,}; division rows: {len(division):,}; summary rows: {len(summary):,}")


if __name__ == "__main__":
    main()
