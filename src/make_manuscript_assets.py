#!/usr/bin/env python3
"""Regenerate manuscript figures, LaTeX tables, macros, and sample data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs"
FIGURES = ROOT / "manuscript/figures"
TABLES = ROOT / "manuscript/tables"
DATA = ROOT / "data"

COLORS = {
    "navy": "#174A7E", "blue": "#4C78A8", "orange": "#F28E2B",
    "green": "#59A14F", "red": "#E15759", "gray": "#7F7F7F",
    "light": "#DDEAF3",
}

MODEL_NAMES = {
    "last": "Persistence",
    "mean4": "Four-week mean",
    "seasonal52": "Seasonal naive",
    "log_growth4": "Log-growth",
    "negative_binomial_glm": "NB-GLM",
    "damped_holt_log": "Damped Holt (log)",
    "baseline_hgb": "Baseline HGB",
    "bio_hgb": "Biology HGB",
    "bio_constrained": "Constrained biology HGB",
    "ensemble_log_hgb": "Log-growth--HGB ensemble",
}


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 160,
        "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def forecast_backtest() -> None:
    predictions = pd.read_csv(
        RESULTS / "dengue_biology_backtest_predictions.csv",
        parse_dates=["target_week_start"],
    )
    selected = pd.read_csv(
        RESULTS / "dengue_biology_selected_models.csv"
    ).set_index("horizon")
    fig, axes = plt.subplots(
        2, 3, figsize=(11.4, 5.8), sharex="col",
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08, "wspace": 0.18},
    )
    for column, horizon in enumerate([1, 2, 4]):
        model = selected.loc[horizon, "selected_model"]
        frame = predictions[
            predictions["horizon"].eq(horizon) & predictions["model"].eq(model)
        ].sort_values("target_week_start")
        top, residual_ax = axes[0, column], axes[1, column]
        top.plot(frame["target_week_start"], frame["actual"], color="black", lw=1.15, label="Observed")
        top.plot(frame["target_week_start"], frame["prediction"], color=COLORS["blue"], lw=1.1, label="Forecast")
        interval = frame.dropna(subset=["lower_80_conformal", "upper_80_conformal"])
        top.fill_between(
            interval["target_week_start"].to_numpy(),
            interval["lower_80_conformal"].to_numpy(float),
            interval["upper_80_conformal"].to_numpy(float),
            color="#9ECAE1", alpha=0.65, linewidth=0, label="80% interval",
        )
        residual = frame["prediction"] - frame["actual"]
        residual_ax.axhline(0, color="#555555", lw=0.7)
        residual_ax.fill_between(
            frame["target_week_start"].to_numpy(), 0, residual.to_numpy(float),
            where=residual.ge(0).to_numpy(), color=COLORS["orange"], alpha=0.75,
        )
        residual_ax.fill_between(
            frame["target_week_start"].to_numpy(), 0, residual.to_numpy(float),
            where=residual.lt(0).to_numpy(), color=COLORS["blue"], alpha=0.75,
        )
        top.set_title(f"{horizon}-week: {MODEL_NAMES.get(model, model)}")
        top.grid(axis="y", color="#E5E5E5", lw=0.5)
        residual_ax.grid(axis="y", color="#E5E5E5", lw=0.5)
        residual_ax.set_xlabel("Target week")
        if column == 0:
            top.set_ylabel("Admissions")
            residual_ax.set_ylabel("Error")
        if column == 2:
            top.legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle("Rolling-origin forecasts and signed residuals", y=0.99, fontsize=11, fontweight="bold")
    save(fig, "forecast_backtest")


def block_bootstrap_skill(group: pd.DataFrame, replicates: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(20261014)
    arrays = []
    years = pd.to_datetime(group["target_week_start"]).dt.year
    for _, year in group.groupby(years):
        arrays.append(year[["model_error", "persistence_error"]].to_numpy(float))
    draws = []
    for _ in range(replicates):
        sample = []
        for values in arrays:
            n = len(values)
            selected = []
            while len(selected) < n:
                start = int(rng.integers(0, n))
                selected.extend(values[(start + offset) % n] for offset in range(4))
            sample.extend(selected[:n])
        draw = np.asarray(sample)
        denominator = draw[:, 1].mean()
        draws.append(1.0 - draw[:, 0].mean() / denominator if denominator else np.nan)
    return float(np.nanquantile(draws, 0.025)), float(np.nanquantile(draws, 0.975))


def forecast_performance() -> None:
    predictions = pd.read_csv(RESULTS / "dengue_biology_backtest_predictions.csv")
    order = list(MODEL_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.9), sharey=True)
    for ax, horizon in zip(axes, [1, 2, 4]):
        horizon_predictions = predictions[predictions["horizon"].eq(horizon)]
        pivot = horizon_predictions.pivot_table(
            index=["target_week_start", "actual"], columns="model", values="prediction"
        ).reset_index()
        y_positions, points, lowers, uppers, colors = [], [], [], [], []
        for position, model in enumerate(order):
            columns = ["target_week_start", "actual", "last"]
            if model != "last":
                columns.append(model)
            paired = pivot[columns].dropna()
            paired["model_error"] = (
                paired[model] - paired["actual"]
            ).abs()
            paired["persistence_error"] = (paired["last"] - paired["actual"]).abs()
            skill = 1.0 - paired["model_error"].mean() / paired["persistence_error"].mean()
            lower, upper = block_bootstrap_skill(paired)
            y_positions.append(position)
            points.append(100 * skill)
            lowers.append(100 * (skill - lower))
            uppers.append(100 * (upper - skill))
            colors.append(
                COLORS["orange"] if model in {"bio_hgb", "bio_constrained"}
                else COLORS["gray"] if model in {"last", "mean4", "seasonal52"}
                else COLORS["blue"]
            )
        ax.axvline(0, color="#555555", lw=0.8)
        for y, point, lower, upper, color in zip(y_positions, points, lowers, uppers, colors):
            ax.errorbar(point, y, xerr=[[lower], [upper]], fmt="o", color=color, ecolor=color, capsize=2, markersize=4)
        ax.set_xscale("symlog", linthresh=20)
        ax.set_title(f"{horizon}-week horizon")
        ax.set_xlabel("MAE skill vs persistence (%)")
        ax.grid(axis="x", color="#E5E5E5", lw=0.6)
        ax.set_yticks(y_positions, [MODEL_NAMES[model] for model in order])
        ax.invert_yaxis()
    axes[0].set_ylabel("Forecast model")
    fig.suptitle("Forecast skill with year-stratified moving-block 95% intervals", y=0.99, fontsize=11, fontweight="bold")
    save(fig, "forecast_performance")


def allocation_frontier() -> None:
    frontier = pd.read_csv(RESULTS / "dengue_phase2_joint_reserve_frontier.csv")
    fixed = frontier[frontier["reserve_rule"].str.startswith("point_plus_")].sort_values("uplift_fraction")
    adaptive = frontier[frontier["reserve_rule"].isin([
        "point_max3_envelope", "upper80_pooled", "upper80_max3_envelope"
    ])]
    summary = pd.read_csv(RESULTS / "dengue_phase2_joint_backtest_summary.csv")
    oracle = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["alignment_scenario"].eq("confirmed_dghs")
        & summary["forecast_level"].eq("observed_total_oracle")
        & summary["policy"].eq("pooled3")
    ].iloc[0]
    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    ax.plot(
        fixed["aggregate_budget_to_actual_ratio"] * 100,
        fixed["mean_shortage_fraction"] * 100,
        marker="o", markersize=3.5, color=COLORS["navy"], lw=1.5,
        label="Fixed uplift on point forecast",
    )
    labels = {
        "point_max3_envelope": ("s", "Point + max-share envelope"),
        "upper80_pooled": ("^", "Upper 80% + pooled"),
        "upper80_max3_envelope": ("D", "Upper 80% + envelope"),
    }
    for _, row in adaptive.iterrows():
        marker, label = labels[row["reserve_rule"]]
        ax.scatter(
            row["aggregate_budget_to_actual_ratio"] * 100,
            row["mean_shortage_fraction"] * 100,
            s=55, marker=marker, color=COLORS["orange"], edgecolor="white",
            linewidth=0.6, zorder=4, label=label,
        )
    ax.scatter(
        oracle["aggregate_budget_to_actual_ratio"] * 100,
        oracle["mean_shortage_fraction"] * 100,
        marker="*", s=95, color=COLORS["green"], zorder=5,
        label="Observed-total oracle + pooled",
    )
    ax.set(
        xlabel="Total allocated units / total observed admissions (%)",
        ylabel="Mean divisional shortage (%)",
        title="Preparedness–shortage trade-off in 101 comparable weeks",
    )
    ax.grid(color="#E5E5E5", lw=0.6)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, "allocation_frontier")


def censoring_robustness() -> None:
    stress = pd.read_csv(RESULTS / "dengue_phase3_censoring_stress_summary.csv")
    frame = stress[stress["budget_uplift_over_point_forecast"].eq(0.5)]
    profiles = ["uniform", "capacity_linked", "inverse_capacity", *[f"rotation_{i}" for i in range(1, 8)]]
    uplifts = [0.0, 0.25, 0.5, 1.0]
    matrix = np.zeros((len(profiles), len(uplifts)))
    for row_index, profile in enumerate(profiles):
        for column, uplift in enumerate(uplifts):
            block = frame[frame["truth_profile"].eq(profile) & frame["latent_uplift_max"].eq(uplift)]
            pooled = block[block["allocation_policy"].eq("pooled_reported")]["mean_shortage_fraction"].iloc[0]
            proxy = block[block["allocation_policy"].eq("capacity_prior_full_weighted")]["mean_shortage_fraction"].iloc[0]
            matrix[row_index, column] = 100 * (proxy - pooled)
    fig, ax = plt.subplots(figsize=(7.7, 5.2))
    limit = max(0.2, float(np.abs(matrix).max()))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for row_index in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row_index, f"{matrix[row_index, column]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(uplifts)), [f"+{int(100*u)}%" for u in uplifts])
    ax.set_yticks(range(len(profiles)), [profile.replace("_", " ") for profile in profiles])
    ax.set_xlabel("Maximum latent-demand uplift")
    ax.set_ylabel("Assumed spatial pattern")
    ax.set_title("Full capacity-prior minus pooled-share shortage\n(percentage points; 50% reserve)")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Shortage difference (percentage points)")
    save(fig, "censoring_robustness")


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 3.2))
    ax.axis("off")
    boxes = [
        (0.01, "Inputs\nDGHS + weather"),
        (0.205, "Forecasts\n1, 2, 4 weeks"),
        (0.40, "Uncertainty\n80% intervals"),
        (0.595, "Decisions\ndivisional reserves"),
        (0.79, "Outputs\nshortage + unused"),
    ]
    for index, (x, label) in enumerate(boxes):
        box = FancyBboxPatch(
            (x, 0.34), 0.17, 0.34, boxstyle="round,pad=0.012",
            facecolor=COLORS["light"], edgecolor=COLORS["navy"], linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(x + 0.085, 0.51, label, ha="center", va="center", fontsize=9)
        if index < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.192, 0.51), xytext=(x + 0.172, 0.51), arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.1})
    ax.text(0.5, 0.88, "Decision-focused forecast evaluation under potentially censored surveillance", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.15, "Information cutoff: end of forecast-origin week", ha="center", fontsize=9, color="#444444")
    save(fig, "study_workflow")


def allocation_weighted_shortage(detail: pd.DataFrame, forecast_level: str, policy: str) -> float:
    block = detail[
        detail["analysis_rule"].eq("published_primary")
        & detail["alignment_scenario"].eq("confirmed_dghs")
        & detail["forecast_level"].eq(forecast_level)
        & detail["policy"].eq(policy)
    ]
    return float(block["shortage_admission_equivalents"].sum() / block["division_actual_admissions"].sum())


def latex_tables_and_macros() -> None:
    selected = pd.read_csv(RESULTS / "dengue_biology_selected_models.csv")
    selected_rows = []
    for _, row in selected.iterrows():
        selected_rows.append(
            f"{int(row['horizon'])} & {MODEL_NAMES.get(row['selected_model'], row['selected_model'])} & "
            f"{int(row['n_common'])} & {row['mae']:.1f} & {row['wape']*100:.1f} & {row['coverage_80_conformal']*100:.1f} & "
            f"{row['coverage_80_conformal_2026']*100:.1f} \\\\"
        )
    (TABLES / "forecast_selected.tex").write_text("\n".join(selected_rows) + "\n", encoding="utf-8")

    metrics = pd.read_csv(RESULTS / "dengue_biology_benchmark_metrics.csv")
    full_rows = []
    for horizon in [1, 2, 4]:
        block = metrics[metrics["horizon"].eq(horizon)].set_index("model")
        for model, label in MODEL_NAMES.items():
            row = block.loc[model]
            full_rows.append(
                f"{horizon} & {label} & {int(row['n_common'])} & {row['mae']:.1f} & {row['rmse']:.1f} & "
                f"{row['wape']*100:.1f} & {row['mean_bias']:.1f} & {row['peak_week_mae']:.1f} & "
                f"{row['mae_skill_vs_persistence']*100:.1f} & {row['coverage_80_conformal']*100:.1f} & "
                f"{row['width_80_conformal']:.1f} & {row['interval_score_80_conformal']:.1f} \\\\"
            )
    (TABLES / "forecast_full.tex").write_text("\n".join(full_rows) + "\n", encoding="utf-8")

    summary = pd.read_csv(RESULTS / "dengue_phase2_joint_backtest_summary.csv")
    detail = pd.read_csv(RESULTS / "dengue_phase2_joint_backtest_detail.csv")
    primary = summary[
        summary["analysis_rule"].eq("published_primary")
        & summary["alignment_scenario"].eq("confirmed_dghs")
    ].copy()
    primary["strategy_key"] = primary["forecast_level"] + "__" + primary["policy"]
    wanted = [
        ("observed_total_oracle__pooled3", "Observed-total oracle + pooled", "observed_total_oracle", "pooled3"),
        ("point__pooled3", "Point + pooled", "point", "pooled3"),
        ("point__max3_envelope", "Point + max-share envelope", "point", "max3_envelope"),
        ("upper80__pooled3", "Upper 80\\% + pooled", "upper80", "pooled3"),
        ("upper80__max3_envelope", "Upper 80\\% + envelope", "upper80", "max3_envelope"),
    ]
    allocation_rows = []
    for key, label, forecast_level, policy in wanted:
        row = primary[primary["strategy_key"].eq(key)].iloc[0]
        allocation_rows.append(
            f"{label} & {row['mean_shortage_fraction']*100:.1f} & {row['demand_weighted_shortage_fraction']*100:.1f} & "
            f"{row['mean_weekly_shortage_admission_equivalents']:.1f} & {row['p90_shortage_fraction']*100:.1f} & "
            f"{row['mean_unused_fraction_of_allocation']*100:.1f} & {row['aggregate_budget_to_actual_ratio']*100:.1f} & "
            f"{row['full_division_coverage_rate']*100:.1f} \\\\"
        )
    (TABLES / "allocation_primary.tex").write_text("\n".join(allocation_rows) + "\n", encoding="utf-8")

    frontier = pd.read_csv(RESULTS / "dengue_phase2_joint_reserve_frontier.csv")
    fixed = frontier[
        frontier["reserve_rule"].str.startswith("point_plus_")
    ].copy()
    fixed = fixed[
        fixed["uplift_fraction"].mul(100).round().astype(int).mod(10).eq(0)
    ].sort_values("uplift_fraction")
    fixed_rows = []
    for _, row in fixed.iterrows():
        fixed_rows.append(
            f"{row['uplift_fraction']*100:.0f} & {row['mean_shortage_fraction']*100:.1f} & "
            f"{row['demand_weighted_shortage_fraction']*100:.1f} & {row['surge_mean_shortage_fraction']*100:.1f} & "
            f"{row['mean_weekly_shortage_admission_equivalents']:.1f} & "
            f"{row['mean_unused_fraction_of_allocation']*100:.1f} & {row['aggregate_budget_to_actual_ratio']*100:.1f} \\\\"
        )
    (TABLES / "fixed_reserve_frontier.tex").write_text("\n".join(fixed_rows) + "\n", encoding="utf-8")

    matched = pd.read_csv(RESULTS / "dengue_phase2_joint_budget_matched_comparisons.csv")
    matched_labels = {
        "point_max3_envelope": "Point + max-share envelope",
        "upper80_pooled": "Upper 80\\% + pooled",
        "upper80_max3_envelope": "Upper 80\\% + envelope",
    }
    matched_rows = []
    for _, row in matched.iterrows():
        matched_rows.append(
            f"{matched_labels[row['adaptive_rule']]} & {row['matched_fixed_uplift_fraction']*100:.1f} & "
            f"{row['adaptive_minus_fixed_mean_shortage']*100:+.2f} & "
            f"[{row['mean_shortage_difference_95pct_ci_lower']*100:+.2f}, {row['mean_shortage_difference_95pct_ci_upper']*100:+.2f}] & "
            f"{row['adaptive_minus_fixed_demand_weighted_shortage']*100:+.2f} & "
            f"[{row['weighted_shortage_difference_95pct_ci_lower']*100:+.2f}, {row['weighted_shortage_difference_95pct_ci_upper']*100:+.2f}] & "
            f"{row['adaptive_minus_fixed_surge_mean_shortage']*100:+.2f} & "
            f"[{row['surge_shortage_difference_95pct_ci_lower']*100:+.2f}, {row['surge_shortage_difference_95pct_ci_upper']*100:+.2f}] & "
            f"{row['adaptive_minus_fixed_mean_unused']*100:+.2f} \\\\"
        )
    (TABLES / "budget_matched.tex").write_text("\n".join(matched_rows) + "\n", encoding="utf-8")

    forecast_sensitivity = pd.read_csv(
        RESULTS / "dengue_phase2_forecast_choice_sensitivity.csv"
    )
    sensitivity_labels = {
        "log_growth4": "Retrospectively best log-growth",
        "ensemble_log_hgb": "Log-growth--HGB ensemble",
        "baseline_hgb": "Pre-specified baseline HGB",
        "rolling_selected_all": "Prospective rolling selector",
        "direct_division_log_growth4": "Direct divisional log-growth",
    }
    policy_labels = {
        "pooled3": "Pooled shares",
        "max3_envelope": "Max-share envelope",
        "direct": "Direct forecasts",
    }
    sensitivity_rows = []
    for _, row in forecast_sensitivity.iterrows():
        sensitivity_rows.append(
            f"{sensitivity_labels.get(row['forecast_model'], row['forecast_model'])} & "
            f"{policy_labels.get(row['allocation_policy'], row['allocation_policy'])} & "
            f"{int(row['target_weeks'])} & {row['mean_shortage_fraction']*100:.1f} & "
            f"{row['demand_weighted_shortage_fraction']*100:.1f} & "
            f"{row['mean_weekly_shortage_admission_equivalents']:.1f} & "
            f"{row['aggregate_budget_to_actual_ratio']*100:.1f} & "
            f"{row['mean_unused_fraction_of_allocation']*100:.1f} \\\\"
        )
    (TABLES / "forecast_choice_sensitivity.tex").write_text(
        "\n".join(sensitivity_rows) + "\n", encoding="utf-8"
    )

    seasons = pd.read_csv(RESULTS / "dengue_biology_benchmark_metrics_by_season.csv")
    season_rows = []
    for year in sorted(seasons["target_year"].unique()):
        for model in ["ensemble_log_hgb", "baseline_hgb"]:
            row = seasons[
                seasons["horizon"].eq(1)
                & seasons["target_year"].eq(year)
                & seasons["model"].eq(model)
            ].iloc[0]
            season_rows.append(
                f"{int(year)} & {MODEL_NAMES[model]} & {int(row['n_common'])} & "
                f"{row['mae']:.1f} & {row['wape']*100:.1f} & {row['mean_bias']:.1f} \\\\"
            )
    (TABLES / "forecast_by_season.tex").write_text("\n".join(season_rows) + "\n", encoding="utf-8")

    current = pd.read_csv(RESULTS / "dengue_biology_current_forecast.csv")
    one = current[current["horizon_weeks"].eq(1)].iloc[0]
    one_selected = selected[selected["horizon"].eq(1)].iloc[0]
    macros = [
        f"\\newcommand{{\\OneWeekForecast}}{{{one['point_forecast_admissions']:,.0f}}}",
        f"\\newcommand{{\\OneWeekLower}}{{{one['lower_80_conformal']:,.0f}}}",
        f"\\newcommand{{\\OneWeekUpper}}{{{one['upper_80_conformal']:,.0f}}}",
        "\\newcommand{\\JointWeeks}{101}",
        f"\\newcommand{{\\ForecastWAPE}}{{{one_selected['wape']*100:.1f}\\%}}",
        f"\\newcommand{{\\ForecastCoverage}}{{{one_selected['coverage_80_conformal']*100:.1f}\\%}}",
    ]
    (TABLES / "result_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def sample_dataset() -> None:
    weekly = pd.read_csv(DATA / "processed/dengue_weekly_model_verified.csv")
    keep = [
        column for column in [
            "week_start", "week_end", "week_year", "week_number", "admissions",
            "target_status", "target_source",
        ] if column in weekly.columns
    ]
    sample = weekly.loc[weekly.get("admissions", pd.Series(index=weekly.index)).notna(), keep].tail(40)
    sample.to_csv(DATA / "sample_weekly_admissions.csv", index=False)


def main() -> None:
    setup()
    forecast_backtest()
    forecast_performance()
    allocation_frontier()
    censoring_robustness()
    workflow()
    latex_tables_and_macros()
    sample_dataset()
    print(f"Wrote manuscript assets to {FIGURES} and {TABLES}")


if __name__ == "__main__":
    main()
