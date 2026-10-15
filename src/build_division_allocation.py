#!/usr/bin/env python3
"""Build a transparent, restartable weekly dengue allocation prototype.

The prototype deliberately uses an abstract admission-service-equivalent unit.
It must not be relabelled as beds, staff, blood products, or money until the
corresponding capacity, conversion, and cost data are supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIVISION = ROOT / "work/weekly_covariates/dengue_weekly_division_published.csv"
INPUT_FORECAST = ROOT / "outputs/dengue_biology_current_forecast.csv"
WORK = ROOT / "work/allocation"
OUTPUTS = ROOT / "outputs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--capacity-budget",
        type=float,
        default=None,
        help="Available weekly admission-service-equivalent units; default is the 1-week point forecast.",
    )
    parser.add_argument("--share-weeks", type=int, default=3)
    parser.add_argument("--partial-week-threshold", type=float, default=0.50)
    return parser.parse_args()


def latest_complete_share_weeks(raw: pd.DataFrame, count: int, threshold: float):
    year = int(raw["selected_year"].max())
    data = raw[raw["selected_year"].eq(year)].copy()
    data["week_number"] = data["published_week_label"].str.extract(r"(\d+)").astype(int)
    divisions = sorted(data["division_label"].dropna().unique())
    pivot = data.pivot_table(
        index=["published_week_label", "week_number"],
        columns="division_label",
        values="admissions",
        aggfunc="sum",
    ).reset_index().sort_values("week_number")
    pivot["division_count"] = pivot[divisions].notna().sum(axis=1)
    pivot["total"] = pivot[divisions].sum(axis=1, min_count=len(divisions))
    pivot["previous_three_median"] = pivot["total"].shift().rolling(3, min_periods=3).median()
    pivot["partial_flag"] = (
        pivot["previous_three_median"].notna()
        & pivot["total"].lt(threshold * pivot["previous_three_median"])
    )
    eligible = pivot[
        pivot["total"].notna()
        & pivot["division_count"].eq(len(divisions))
        & ~pivot["partial_flag"]
    ].tail(count)
    if len(eligible) != count:
        raise RuntimeError(f"Needed {count} complete share weeks; found {len(eligible)}")
    shares = eligible[divisions].div(eligible["total"], axis=0)
    shares.index = eligible["published_week_label"].tolist()
    audit = pivot[
        [
            "published_week_label",
            "week_number",
            "total",
            "division_count",
            "previous_three_median",
            "partial_flag",
        ]
    ].copy()
    audit["used_for_scenario_share"] = audit["published_week_label"].isin(shares.index)
    return year, divisions, shares, audit


def build_scenarios(forecast: pd.Series, shares: pd.DataFrame) -> pd.DataFrame:
    totals = {
        "lower": float(forecast["lower_80_conformal"]),
        "point": float(forecast["point_forecast_admissions"]),
        "upper": float(forecast["upper_80_conformal"]),
    }
    records = []
    for level, total in totals.items():
        for week, row in shares.iterrows():
            for division, share in row.items():
                records.append(
                    {
                        "scenario_id": f"{level}_{week}",
                        "national_forecast_level": level,
                        "share_source_week": week,
                        "division": division,
                        "division_share": float(share),
                        "national_admissions": total,
                        "division_demand_units": total * float(share),
                        "scenario_probability": 1.0 / (3 * len(shares)),
                    }
                )
    return pd.DataFrame(records)


def optimize(scenarios: pd.DataFrame, divisions: list[str], budget: float):
    scenario_ids = scenarios["scenario_id"].drop_duplicates().tolist()
    demand = scenarios.pivot(index="scenario_id", columns="division", values="division_demand_units").loc[
        scenario_ids, divisions
    ].to_numpy()
    k_count, d_count = demand.shape
    x_start = 0
    s_start = d_count
    u_index = d_count + k_count * d_count
    n_variables = u_index + 1
    a_ub, b_ub = [], []
    for k in range(k_count):
        for d in range(d_count):
            shortage_index = s_start + k * d_count + d
            row = np.zeros(n_variables)
            row[x_start + d] = -1.0
            row[shortage_index] = -1.0
            a_ub.append(row)
            b_ub.append(-demand[k, d])

            row = np.zeros(n_variables)
            row[shortage_index] = 1.0
            row[u_index] = -demand[k, d]
            a_ub.append(row)
            b_ub.append(0.0)
    budget_row = np.zeros(n_variables)
    budget_row[:d_count] = 1.0
    a_ub.append(budget_row)
    b_ub.append(budget)
    bounds = [(0.0, float(demand[:, d].max())) for d in range(d_count)]
    bounds += [(0.0, None)] * (k_count * d_count) + [(0.0, 1.0)]

    # Lexicographic objective: first minimize the worst divisional shortage
    # fraction, then minimize equally weighted expected shortage without making
    # that fairness result worse.
    stage_one_objective = np.zeros(n_variables)
    stage_one_objective[u_index] = 1.0
    first = linprog(
        stage_one_objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not first.success:
        raise RuntimeError(first.message)
    fairness_limit = float(first.x[u_index]) + 1e-9
    fairness_row = np.zeros(n_variables)
    fairness_row[u_index] = 1.0
    a_ub.append(fairness_row)
    b_ub.append(fairness_limit)
    second_objective = np.zeros(n_variables)
    second_objective[s_start:u_index] = 1.0 / k_count
    second = linprog(
        second_objective,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        bounds=bounds,
        method="highs",
    )
    if not second.success:
        raise RuntimeError(second.message)
    return second.x[:d_count], fairness_limit


def evaluate(allocation: np.ndarray, scenarios: pd.DataFrame, divisions: list[str]):
    frame = scenarios.copy()
    allocation_map = dict(zip(divisions, allocation))
    frame["allocation_units"] = frame["division"].map(allocation_map)
    frame["shortage_units"] = (
        frame["division_demand_units"] - frame["allocation_units"]
    ).clip(lower=0.0)
    frame["service_fraction"] = np.minimum(
        1.0,
        frame["allocation_units"] / frame["division_demand_units"].replace(0, np.nan),
    )
    return frame


def main() -> None:
    args = arguments()
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_DIVISION)
    forecast_table = pd.read_csv(INPUT_FORECAST)
    forecast = forecast_table[forecast_table["horizon_weeks"].eq(1)].iloc[0]
    year, divisions, shares, week_audit = latest_complete_share_weeks(
        raw, args.share_weeks, args.partial_week_threshold
    )
    scenarios = build_scenarios(forecast, shares)
    budget = (
        float(args.capacity_budget)
        if args.capacity_budget is not None
        else float(forecast["point_forecast_admissions"])
    )
    robust, fairness_limit = optimize(scenarios, divisions, budget)
    pooled = shares.sum(axis=0) / shares.sum(axis=0).sum()
    proportional = pooled.loc[divisions].to_numpy() * budget
    robust_eval = evaluate(robust, scenarios, divisions)
    proportional_eval = evaluate(proportional, scenarios, divisions)

    rows = []
    for index, division in enumerate(divisions):
        r = robust_eval[robust_eval["division"].eq(division)]
        p = proportional_eval[proportional_eval["division"].eq(division)]
        rows.append(
            {
                "division": division,
                "robust_allocation_units": robust[index],
                "robust_allocation_share": robust[index] / budget,
                "pooled_share_allocation_units": proportional[index],
                "pooled_recent_admission_share": pooled[division],
                "robust_worst_service_fraction": r["service_fraction"].min(),
                "pooled_worst_service_fraction": p["service_fraction"].min(),
                "robust_expected_shortage_units": r["shortage_units"].mean(),
                "pooled_expected_shortage_units": p["shortage_units"].mean(),
            }
        )
    allocation = pd.DataFrame(rows).sort_values("robust_allocation_units", ascending=False)

    scenario_summary = (
        robust_eval.groupby(
            ["scenario_id", "national_forecast_level", "share_source_week", "national_admissions"],
            as_index=False,
        )
        .agg(
            total_robust_shortage_units=("shortage_units", "sum"),
            worst_robust_service_fraction=("service_fraction", "min"),
        )
    )
    proportional_summary = (
        proportional_eval.groupby("scenario_id", as_index=False)
        .agg(
            total_pooled_shortage_units=("shortage_units", "sum"),
            worst_pooled_service_fraction=("service_fraction", "min"),
        )
    )
    scenario_summary = scenario_summary.merge(proportional_summary, on="scenario_id")
    assumptions = pd.DataFrame(
        [
            ("forecast_horizon_weeks", 1, "The allocation is for the next reported week."),
            ("capacity_budget_units", budget, "Defaults to the national point forecast; override on the command line."),
            ("resource_unit", "admission-service-equivalent", "Abstract placeholder, not a bed, worker, product, or currency unit."),
            ("share_scenario_weeks", args.share_weeks, "Latest completed DGHS division week labels used as spatial scenarios."),
            ("partial_week_threshold", args.partial_week_threshold, "A latest week below this fraction of the previous-three-week median is excluded."),
            ("forecast_interval", "planning-adjusted nominal 80%", "Lower/point/upper national totals define demand scenarios."),
            ("scenario_weights", "equal", "All three national levels crossed with all recent spatial-share weeks."),
            ("fairness_rule", "lexicographic max-min", "First minimizes worst proportional divisional shortage, then expected shortage."),
        ],
        columns=["assumption", "value", "meaning"],
    )

    allocation.to_csv(OUTPUTS / "dengue_division_allocation_prototype.csv", index=False)
    scenarios.to_csv(OUTPUTS / "dengue_division_allocation_scenarios.csv", index=False)
    scenario_summary.to_csv(OUTPUTS / "dengue_division_allocation_scenario_summary.csv", index=False)
    assumptions.to_csv(OUTPUTS / "dengue_division_allocation_assumptions.csv", index=False)
    week_audit.to_csv(OUTPUTS / "dengue_division_week_completeness_audit.csv", index=False)

    used_weeks = ", ".join(shares.index)
    partial = week_audit[week_audit["partial_flag"]]["published_week_label"].tolist()
    report = f"""# Weekly dengue division allocation prototype

Generated: {datetime.now().astimezone().isoformat()}

## What is now possible

The pipeline can demonstrate a one-week, uncertainty-aware allocation across
Bangladesh's eight divisions. It uses the national forecast interval and the
observed variation in recent division shares. The optimization first minimizes
the worst proportional divisional shortage and then minimizes expected total
shortage without weakening that fairness result.

This is a **research prototype, not an operational allocation order**. The
resource is an abstract weekly admission-service-equivalent. The default budget
is {budget:,.1f} units, equal to the national point forecast. It must be replaced
with a real resource budget and conversion rule before policy use.

## Inputs and scenarios

- Target week: {forecast['target_week_start']} through {forecast['target_week_end']}.
- National lower / point / upper totals: {forecast['lower_80_conformal']:,.1f} /
  {forecast['point_forecast_admissions']:,.1f} / {forecast['upper_80_conformal']:,.1f}.
- Division-share scenarios: DGHS {year} raw labels {used_weeks}.
- Automatically excluded partial-looking labels: {', '.join(partial) if partial else 'none'}.
- Total demand scenarios: {scenarios['scenario_id'].nunique()} (three national totals by
  {len(shares)} recent division-share patterns).

## Prototype result

- Worst service fraction protected by the robust allocation: {1-fairness_limit:.1%}.
- Worst service fraction under pooled-share allocation: {scenario_summary['worst_pooled_service_fraction'].min():.1%}.
- Mean robust shortage across scenarios: {scenario_summary['total_robust_shortage_units'].mean():,.1f} units.
- Mean pooled-share shortage across scenarios: {scenario_summary['total_pooled_shortage_units'].mean():,.1f} units.

This exposes the intended trade-off: the robust solution raises the minimum
division-level service fraction, while the pooled-share rule has slightly lower
mean shortage. Which objective is appropriate is a policy choice, not a fact the
forecast can determine.

The allocation table is `outputs/dengue_division_allocation_prototype.csv`.
It contains both the robust result and a simple allocation proportional to the
pooled admissions in the three recent completed weeks.

## Blocking data before operational use

1. The actual resource and national quantity: staffed beds, nurses, IV fluids,
   test kits, referral slots, cash, or another specified item.
2. Current stock/capacity and committed baseline demand by division (preferably
   hospital), including lead times and transfer limits.
3. A conversion between forecast admissions and resource need, including length
   of stay and severity/ICU probability where applicable.
4. Costs, shortage penalties, minimum service rules, and policy priorities
   approved by decision makers.
5. Confirmation that the latest Sunday-Saturday DGHS week is complete. The
   automatic partial-week rule is only a safeguard.

## Reproduce or change the budget

Run `python3 work/allocation/build_division_allocation.py`. To test a real
quantity, add `--capacity-budget NUMBER`. The process is deterministic and
rewrites the five allocation outputs plus the checkpoint manifest.
"""
    (OUTPUTS / "dengue_division_allocation_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            str(INPUT_DIVISION.relative_to(ROOT)): sha256(INPUT_DIVISION),
            str(INPUT_FORECAST.relative_to(ROOT)): sha256(INPUT_FORECAST),
        },
        "parameters": {
            "capacity_budget": budget,
            "share_weeks": args.share_weeks,
            "partial_week_threshold": args.partial_week_threshold,
        },
        "selected_share_labels": list(shares.index),
        "excluded_partial_labels": partial,
        "outputs": [
            "outputs/dengue_division_allocation_prototype.csv",
            "outputs/dengue_division_allocation_scenarios.csv",
            "outputs/dengue_division_allocation_scenario_summary.csv",
            "outputs/dengue_division_allocation_assumptions.csv",
            "outputs/dengue_division_week_completeness_audit.csv",
            "outputs/dengue_division_allocation_report.md",
        ],
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint = f"""# Division allocation checkpoint

Status: complete
Updated: {manifest['generated_at']}

Resume with `python3 work/allocation/build_division_allocation.py`.
The default capacity budget is the current one-week national point forecast.
Pass `--capacity-budget NUMBER` to replace the abstract demonstration budget.
"""
    (WORK / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")
    print(allocation.to_string(index=False))
    print(json.dumps({"status": "complete", "budget": budget, "share_weeks": list(shares.index)}, indent=2))


if __name__ == "__main__":
    main()
