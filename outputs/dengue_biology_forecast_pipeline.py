#!/usr/bin/env python3
"""Compare baseline and biology-informed weekly dengue admission forecasts.

The experiment is leakage-safe and restartable. Biology enters through an
explicit Ae. aegypti temperature-suitability curve and lagged weather summaries.
The constrained model enforces nonnegative Poisson forecasts and monotonicity
with respect to the engineered thermal-suitability and heat-stress features.
It does not impose the scientifically invalid rule that raw temperature or rain
must always increase reported dengue admissions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "work/expanded_weekly/dengue_weekly_model_verified.csv"
DAILY_WEATHER = PROJECT / "work/weekly_covariates/weather_daily_proxy.csv"
WORK = PROJECT / "work/biology_forecast"
RESULTS = WORK / "results"
OUTPUTS = PROJECT / "outputs"
MANIFEST = WORK / "manifest.json"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning, message="Downcasting object dtype arrays"
)


@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 2, 4)
    evaluation_start: str = "2024-01-01"
    minimum_training_rows: int = 80
    calibration_minimum: int = 20
    interval_alpha: float = 0.20
    interval_window: int = 52
    interval_safety_factor: float = 1.10
    random_state: int = 20261012
    max_iter: int = 120
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 12
    l2_regularization: float = 1.0
    thermal_min_c: float = 17.8
    thermal_optimum_c: float = 29.1
    thermal_max_c: float = 34.6
    nb_l2_regularization: float = 1.00
    nb_max_iter: int = 100
    rolling_selection_minimum: int = 20
    bootstrap_replicates: int = 2000
    bootstrap_block_length: int = 4


CONFIG = Config()
POINT_MODELS = [
    "last", "mean4", "seasonal52", "log_growth4", "negative_binomial_glm",
    "damped_holt_log", "baseline_hgb", "bio_hgb", "bio_constrained",
    "ensemble_log_hgb", "rolling_selected_hgb", "rolling_selected_all",
]
COMMON_TARGET_MODELS = [
    "last", "mean4", "seasonal52", "log_growth4", "negative_binomial_glm",
    "baseline_hgb", "bio_hgb", "bio_constrained",
]
ML_MODELS = ["baseline_hgb", "bio_hgb", "bio_constrained"]
MODEL_SELECTION_CANDIDATES = [
    "last", "mean4", "seasonal52", "log_growth4", "negative_binomial_glm",
    "damped_holt_log", "ensemble_log_hgb", *ML_MODELS,
]
TARGET_LAGS = [0, 1, 2, 3, 4, 8, 13, 26, 52]
ROLL_WINDOWS = [4, 8, 13, 26]
WEATHER_LAGS = [0, 1, 2, 4, 8]
BIO_LAGS = [0, 1, 2, 3, 4, 6, 8, 10, 12]
RAW_WEATHER = [
    "proxy_mean_precip_sum_mm",
    "proxy_mean_rainy_days_ge_1mm",
    "proxy_mean_temp_mean_c",
    "proxy_mean_rh_mean_pct",
]
BIO_COLUMNS = [
    "bio_thermal_suitability",
    "bio_suitable_observation_fraction",
    "bio_hot_stress_c",
    "bio_heavy_rain_fraction",
    "bio_vector_pressure_proxy",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash() -> str:
    return hashlib.sha256(json.dumps(asdict(CONFIG), sort_keys=True).encode()).hexdigest()


def save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = datetime.now().astimezone().isoformat()
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def initialize_manifest() -> dict:
    fingerprint = {
        "input_sha256": sha256(INPUT),
        "daily_weather_sha256": sha256(DAILY_WEATHER),
        "pipeline_sha256": sha256(Path(__file__)),
        "config_sha256": config_hash(),
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in fingerprint.items()):
            return old
    manifest = {
        "objective": "Rolling-origin comparison of baseline and biology-informed weekly dengue forecasts",
        **fingerprint,
        "completed_horizons": [],
        "completed_stages": [],
        "status": "initialized",
    }
    save_manifest(manifest)
    return manifest


def thermal_suitability(temperature: pd.Series) -> pd.Series:
    """Transparent triangular approximation to the published thermal window.

    This is a sensitivity feature, not a claim that the exact R0(T) curve is
    triangular. It is zero at the published posterior-mean limits and one at
    the published optimum from Mordecai et al. (2017).
    """
    t = pd.to_numeric(temperature, errors="coerce")
    score = pd.Series(np.nan, index=t.index, dtype=float)
    observed = t.notna()
    score.loc[observed] = 0.0
    rising = observed & t.between(CONFIG.thermal_min_c, CONFIG.thermal_optimum_c, inclusive="both")
    falling = observed & t.gt(CONFIG.thermal_optimum_c) & t.le(CONFIG.thermal_max_c)
    score.loc[rising] = (
        (t.loc[rising] - CONFIG.thermal_min_c)
        / (CONFIG.thermal_optimum_c - CONFIG.thermal_min_c)
    )
    score.loc[falling] = (
        (CONFIG.thermal_max_c - t.loc[falling])
        / (CONFIG.thermal_max_c - CONFIG.thermal_optimum_c)
    )
    return score.clip(0.0, 1.0)


def build_weekly_biology() -> pd.DataFrame:
    weather = pd.read_csv(DAILY_WEATHER, parse_dates=["date", "week_start", "week_end"])
    weather["thermal_suitability"] = thermal_suitability(weather["temp_mean_c"])
    weather["suitable_observation"] = weather["thermal_suitability"].gt(0).where(
        weather["thermal_suitability"].notna()
    )
    weather["hot_stress_c"] = (weather["temp_max_c"] - CONFIG.thermal_max_c).clip(lower=0)
    weather["heavy_rain"] = weather["precip_mm"].ge(50).where(weather["precip_mm"].notna())
    humidity_factor = (weather["rh_mean_pct"] / 100).clip(0, 1)
    rainfall_factor = np.log1p(weather["precip_mm"].clip(lower=0))
    weather["vector_pressure_proxy"] = weather["thermal_suitability"] * np.sqrt(humidity_factor) * rainfall_factor

    grouped = weather.groupby("week_start", as_index=False).agg(
        bio_weather_observations=("temp_mean_c", "count"),
        bio_thermal_suitability=("thermal_suitability", "mean"),
        bio_suitable_observation_fraction=("suitable_observation", "mean"),
        bio_hot_stress_c=("hot_stress_c", "mean"),
        bio_heavy_rain_fraction=("heavy_rain", "mean"),
        bio_vector_pressure_proxy=("vector_pressure_proxy", "mean"),
    )
    grouped["bio_weather_complete"] = grouped["bio_weather_observations"].eq(56)
    return grouped


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(INPUT, parse_dates=["week_start", "week_end"])
    frame = frame.sort_values("week_start").reset_index(drop=True)
    expected = pd.date_range(frame.week_start.min(), frame.week_start.max(), freq="W-SUN")
    if not frame.week_start.equals(pd.Series(expected, name="week_start")):
        raise ValueError("Weekly input is not a complete Sunday-start sequence")
    biology = build_weekly_biology()
    frame = frame.merge(biology, on="week_start", how="left")
    for column in ["target_complete", "target_available", "weather_complete"]:
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["observed_target"] = frame["admissions"].where(frame["target_complete"])
    frame["row_index"] = np.arange(len(frame))
    return frame


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], dict[str, int]]:
    features = df[["row_index", "week_start", "week_end"]].copy()
    baseline: list[str] = []
    biology: list[str] = []
    directions: dict[str, int] = {}
    target = df["observed_target"]

    for lag in TARGET_LAGS:
        name = f"target_lag_{lag}"
        features[name] = target.shift(lag)
        features[f"{name}_missing"] = features[name].isna().astype(int)
        baseline.extend([name, f"{name}_missing"])

    for window in ROLL_WINDOWS:
        rolling = target.rolling(window=window, min_periods=1)
        for statistic, values in [
            ("mean", rolling.mean()),
            ("std", rolling.std(ddof=0)),
            ("count", rolling.count()),
        ]:
            name = f"target_roll_{statistic}_{window}"
            features[name] = values
            baseline.append(name)

    valid_weather = df[RAW_WEATHER].where(df["weather_complete"], np.nan)
    for variable in RAW_WEATHER:
        for lag in WEATHER_LAGS:
            name = f"{variable}_lag_{lag}"
            features[name] = valid_weather[variable].shift(lag)
            features[f"{name}_missing"] = features[name].isna().astype(int)
            baseline.extend([name, f"{name}_missing"])

    bio_valid = df[BIO_COLUMNS].where(df["bio_weather_complete"].fillna(False), np.nan)
    for variable in BIO_COLUMNS:
        for lag in BIO_LAGS:
            name = f"{variable}_lag_{lag}"
            features[name] = bio_valid[variable].shift(lag)
            features[f"{name}_missing"] = features[name].isna().astype(int)
            biology.extend([name, f"{name}_missing"])
            if variable in {"bio_thermal_suitability", "bio_suitable_observation_fraction"}:
                directions[name] = 1
            elif variable == "bio_hot_stress_c":
                directions[name] = -1

    features["origin_week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52.1775)
    features["origin_week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52.1775)
    features["time_index"] = df["row_index"]
    baseline.extend(["origin_week_sin", "origin_week_cos", "time_index"])
    return features, baseline, biology, directions


def horizon_frame(
    df: pd.DataFrame,
    base: pd.DataFrame,
    baseline: list[str],
    biology: list[str],
    horizon: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = base.copy()
    target_week = df["week_start"].shift(-horizon)
    target_week_number = df["week_number"].shift(-horizon).astype("Float64")
    frame["target_week_sin"] = np.sin(2 * np.pi * target_week_number / 52.1775)
    frame["target_week_cos"] = np.cos(2 * np.pi * target_week_number / 52.1775)
    frame["target"] = df["observed_target"].shift(-horizon)
    frame["target_complete_future"] = df["target_complete"].shift(-horizon).fillna(False).astype(bool)
    frame["target_week_start"] = target_week
    frame["target_index"] = df["row_index"].shift(-horizon)
    frame["origin_eligible"] = df["target_complete"]
    baseline_h = [*baseline, "target_week_sin", "target_week_cos"]
    return frame, baseline_h, [*baseline_h, *biology]


def hgb(monotonic: list[int] | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=CONFIG.learning_rate,
        max_iter=CONFIG.max_iter,
        max_leaf_nodes=CONFIG.max_leaf_nodes,
        min_samples_leaf=CONFIG.min_samples_leaf,
        l2_regularization=CONFIG.l2_regularization,
        random_state=CONFIG.random_state,
        early_stopping=False,
        monotonic_cst=monotonic,
    )


def median_pipeline(monotonic: list[int] | None = None) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
        ("model", hgb(monotonic)),
    ])


def negative_binomial_prediction(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_target: pd.DataFrame,
) -> tuple[float, float, bool, str]:
    """Fit a regularized negative-binomial GLM with a log link.

    Dispersion is estimated by a training-only method-of-moments calculation.
    Predictors are median-imputed and standardized using the training fold only.
    This supplies a transparent count-regression baseline without adding a new
    external dependency.
    """
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_array = scaler.fit_transform(imputer.fit_transform(x_train))
    target_array = scaler.transform(imputer.transform(x_target))
    design = np.column_stack([np.ones(len(train_array)), train_array])
    target_design = np.column_stack([np.ones(len(target_array)), target_array])
    y = y_train.to_numpy(dtype=float)
    mean_y = max(float(np.mean(y)), 1e-6)
    variance_y = float(np.var(y, ddof=1)) if len(y) > 1 else mean_y
    alpha = float(np.clip((variance_y - mean_y) / (mean_y ** 2), 1e-4, 5.0))
    beta = np.zeros(design.shape[1], dtype=float)
    beta[0] = math.log(mean_y)
    penalty = np.eye(design.shape[1]) * CONFIG.nb_l2_regularization
    penalty[0, 0] = 0.0
    converged = False
    for iteration in range(CONFIG.nb_max_iter):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eta_raw = design @ beta
            if not np.isfinite(eta_raw).all():
                break
            eta = np.clip(eta_raw, -12.0, 16.0)
            mu = np.exp(eta)
            weights = np.clip(mu / (1.0 + alpha * mu), 1e-8, None)
            working = eta + (y - mu) / np.clip(mu, 1e-8, None)
            normal_matrix = design.T @ (weights[:, None] * design) + penalty
            normal_rhs = design.T @ (weights * working)
        if not np.isfinite(normal_matrix).all() or not np.isfinite(normal_rhs).all():
            break
        updated = np.linalg.lstsq(normal_matrix, normal_rhs, rcond=1e-10)[0]
        if not np.isfinite(updated).all():
            break
        updated = np.clip(updated, -12.0, 12.0)
        updated[0] = np.clip(updated[0], -12.0, 16.0)
        if float(np.max(np.abs(updated - beta))) < 1e-7:
            beta = updated
            converged = True
            break
        # Damping improves stability during steep outbreak transitions.
        beta = 0.5 * beta + 0.5 * updated
    eta_target = float(np.clip(target_design @ beta, -12.0, 16.0)[0])
    prediction = max(0.0, float(np.exp(eta_target)))
    message = "IRLS converged" if converged else "IRLS reached maximum iterations"
    return prediction, alpha, converged, message


def log_growth_prediction(recent: pd.Series, horizon: int) -> float:
    """Extrapolate the four-week log-count slope as a classical trend baseline."""
    if len(recent) != 4 or recent.isna().any():
        return np.nan
    values = np.log1p(recent.to_numpy(dtype=float))
    slope, _ = np.polyfit(np.arange(4, dtype=float), values, 1)
    return max(0.0, float(np.expm1(values[-1] + horizon * slope)))


def damped_holt_log_prediction(history: pd.Series, horizon: int) -> float:
    """Training-only damped Holt ETS forecast on the log-count scale.

    The smoothing level, trend weight, and damping parameter are selected by
    one-step-ahead squared error within the available history.  Missing values
    are not imputed: only the most recent contiguous observed segment is used.
    """
    values = pd.to_numeric(history, errors="coerce")
    if values.notna().sum() < 12:
        return np.nan
    last_missing = np.flatnonzero(values.isna().to_numpy())
    if len(last_missing):
        values = values.iloc[last_missing[-1] + 1:]
    if len(values) < 12:
        return np.nan
    y = np.log1p(values.to_numpy(dtype=float))
    best: tuple[float, float, float, float, float] | None = None
    for alpha in (0.2, 0.4, 0.6, 0.8):
        for beta in (0.05, 0.2, 0.4):
            for phi in (0.80, 0.90, 0.98, 1.00):
                level = y[0]
                trend = y[1] - y[0]
                errors: list[float] = []
                for observed in y[1:]:
                    fitted = level + phi * trend
                    errors.append(observed - fitted)
                    new_level = alpha * observed + (1.0 - alpha) * fitted
                    new_trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
                    level, trend = new_level, new_trend
                sse = float(np.dot(errors, errors))
                candidate = (sse, level, trend, phi, alpha)
                if best is None or candidate[0] < best[0]:
                    best = candidate
    assert best is not None
    _, level, trend, phi, _ = best
    multiplier = sum(phi ** step for step in range(1, horizon + 1))
    return max(0.0, float(np.expm1(level + multiplier * trend)))


def predict_horizon(
    df: pd.DataFrame,
    base: pd.DataFrame,
    baseline: list[str],
    biology: list[str],
    directions: dict[str, int],
    horizon: int,
) -> pd.DataFrame:
    frame, baseline_cols, bio_cols = horizon_frame(df, base, baseline, biology, horizon)
    classical_count_cols = [
        "target_lag_0", "target_lag_1", "target_lag_2", "target_lag_4",
        "target_lag_8", "target_lag_13", "target_lag_52",
        "target_roll_mean_4", "target_roll_mean_13", "target_roll_mean_26",
    ]
    classical_cols = [*classical_count_cols, "target_week_sin", "target_week_cos"]
    eval_rows = frame.index[
        frame["origin_eligible"]
        & frame["target_complete_future"]
        & frame["target"].notna()
        & frame["week_start"].ge(pd.Timestamp(CONFIG.evaluation_start))
    ]
    monotonic = [directions.get(column, 0) for column in bio_cols]
    records: list[dict] = []

    for i in eval_rows:
        train_mask = (
            frame["origin_eligible"]
            & frame["target_complete_future"]
            & frame["target"].notna()
            & frame["row_index"].le(i - horizon)
        )
        train = frame.loc[train_mask]
        if len(train) < CONFIG.minimum_training_rows:
            continue
        y_train = train["target"].astype(float)

        baseline_model = median_pipeline()
        baseline_model.fit(train[baseline_cols], y_train)
        bio_model = median_pipeline()
        bio_model.fit(train[bio_cols], y_train)
        constrained_model = median_pipeline(monotonic)
        constrained_model.fit(train[bio_cols], y_train)
        nb_train = train[classical_cols].copy()
        nb_target = frame.loc[[i], classical_cols].copy()
        nb_train[classical_count_cols] = np.log1p(
            nb_train[classical_count_cols].clip(lower=0)
        )
        nb_target[classical_count_cols] = np.log1p(
            nb_target[classical_count_cols].clip(lower=0)
        )
        nb_prediction, nb_alpha, nb_converged, nb_message = negative_binomial_prediction(
            nb_train, y_train, nb_target
        )

        target_index = int(frame.at[i, "target_index"])
        recent = df.loc[max(0, i - 3):i, "observed_target"]
        seasonal_index = target_index - 52
        seasonal = np.nan
        if seasonal_index >= 0 and bool(df.at[seasonal_index, "target_complete"]):
            seasonal = float(df.at[seasonal_index, "observed_target"])
        candidates = {
            "last": float(df.at[i, "observed_target"]),
            "mean4": float(recent.mean()) if recent.notna().sum() == 4 else np.nan,
            "seasonal52": seasonal,
            "log_growth4": log_growth_prediction(recent, horizon),
            "damped_holt_log": damped_holt_log_prediction(
                df.loc[:i, "observed_target"], horizon
            ),
            "negative_binomial_glm": nb_prediction,
            "baseline_hgb": max(0.0, float(baseline_model.predict(frame.loc[[i], baseline_cols])[0])),
            "bio_hgb": max(0.0, float(bio_model.predict(frame.loc[[i], bio_cols])[0])),
            "bio_constrained": max(0.0, float(constrained_model.predict(frame.loc[[i], bio_cols])[0])),
        }
        candidates["ensemble_log_hgb"] = 0.5 * (
            candidates["log_growth4"] + candidates["baseline_hgb"]
        )
        for model, prediction in candidates.items():
            records.append({
                "horizon": horizon,
                "model": model,
                "origin_index": int(i),
                "origin_week_start": frame.at[i, "week_start"].date().isoformat(),
                "target_index": target_index,
                "target_week_start": frame.at[i, "target_week_start"].date().isoformat(),
                "actual": float(frame.at[i, "target"]),
                "prediction": prediction,
                "training_rows": int(len(train)),
                "negative_binomial_dispersion_alpha": (
                    nb_alpha if model == "negative_binomial_glm" else np.nan
                ),
                "fit_converged": (
                    nb_converged if model == "negative_binomial_glm" else True
                ),
                "fit_message": (
                    nb_message if model == "negative_binomial_glm" else ""
                ),
            })
    return pd.DataFrame(records)


def add_rolling_model_selection(predictions: pd.DataFrame) -> pd.DataFrame:
    """Append a prospectively selected HGB prediction at each historical origin.

    Candidate choice uses only earlier candidate forecasts whose targets had
    resolved by the current origin. Before the minimum comparison history is
    available, the ordinary baseline HGB is used by design.
    """
    selected_rows: list[pd.Series] = []
    strategies = [
        ("rolling_selected_hgb", ML_MODELS),
        ("rolling_selected_all", MODEL_SELECTION_CANDIDATES),
    ]
    for strategy_name, strategy_models in strategies:
        candidates = predictions[predictions["model"].isin(strategy_models)].copy()
        for horizon, horizon_data in candidates.groupby("horizon", sort=True):
            for origin in sorted(horizon_data["origin_index"].unique()):
                current = horizon_data[
                    horizon_data["origin_index"].eq(origin)
                ].set_index("model")
                history = horizon_data[
                    horizon_data["origin_index"].lt(origin)
                    & horizon_data["target_index"].le(origin)
                ].copy()
                counts = (
                    history.groupby("model")["prediction"].count()
                    .reindex(strategy_models).fillna(0)
                )
                if (counts >= CONFIG.rolling_selection_minimum).all():
                    history["absolute_error"] = (
                        history["prediction"] - history["actual"]
                    ).abs()
                    choice = (
                        history.groupby("model")["absolute_error"].mean()
                        .reindex(strategy_models).sort_values(kind="stable").index[0]
                    )
                    basis_n = int(counts.min())
                else:
                    choice = "baseline_hgb"
                    basis_n = int(counts.min())
                row = current.loc[choice].copy()
                row["model"] = strategy_name
                row["selected_component_model"] = choice
                row["selection_history_n"] = basis_n
                selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    if selected.empty:
        return predictions
    selected["horizon"] = selected["horizon"].astype(int)
    combined = predictions.copy()
    combined["selected_component_model"] = ""
    combined["selection_history_n"] = np.nan
    return pd.concat([combined, selected], ignore_index=True, sort=False)


def add_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    alpha = CONFIG.interval_alpha
    result: list[pd.DataFrame] = []
    for (_, _), group in predictions.groupby(["horizon", "model"], sort=True):
        group = group.sort_values("origin_index").copy()
        group["residual"] = group["actual"] - group["prediction"]
        lower_asym, upper_asym, lower_conf, upper_conf, counts = [], [], [], [], []
        for row in group.itertuples():
            resolved = group[
                group["target_index"].le(row.origin_index)
                & group["origin_index"].lt(row.origin_index)
                & group["residual"].notna()
            ].tail(CONFIG.interval_window)
            n = len(resolved)
            counts.append(n)
            if n < CONFIG.calibration_minimum or pd.isna(row.prediction):
                lower_asym.append(np.nan); upper_asym.append(np.nan)
                lower_conf.append(np.nan); upper_conf.append(np.nan)
                continue
            residual_values = resolved["residual"]
            lower_asym.append(max(0.0, row.prediction + residual_values.quantile(alpha / 2)))
            upper_asym.append(max(0.0, row.prediction + residual_values.quantile(1 - alpha / 2)))
            rank_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
            # Weekly admission variance grows with the epidemic level.  A square-root
            # scale is variance-stabilising for count data, so calibrate the absolute
            # error after dividing by sqrt(point forecast), then restore that scale at
            # prediction time.  The 52-week window adapts to current error behaviour.
            scores = residual_values.abs() / np.sqrt(resolved["prediction"].clip(lower=1.0))
            radius = (
                float(scores.quantile(rank_level, interpolation="higher"))
                * math.sqrt(max(float(row.prediction), 1.0))
                * CONFIG.interval_safety_factor
            )
            lower_conf.append(max(0.0, row.prediction - radius))
            upper_conf.append(row.prediction + radius)
        group["interval_calibration_n"] = counts
        group["lower_80_asymmetric"] = lower_asym
        group["upper_80_asymmetric"] = upper_asym
        group["lower_80_conformal"] = lower_conf
        group["upper_80_conformal"] = upper_conf
        for suffix in ["asymmetric", "conformal"]:
            lower = group[f"lower_80_{suffix}"]
            upper = group[f"upper_80_{suffix}"]
            group[f"covered_80_{suffix}"] = (
                group["actual"].ge(lower) & group["actual"].le(upper)
            ).where(lower.notna() & upper.notna())
            group[f"width_80_{suffix}"] = upper - lower
            group[f"interval_score_80_{suffix}"] = (
                upper - lower
                + (2 / alpha) * (lower - group["actual"]).clip(lower=0)
                + (2 / alpha) * (group["actual"] - upper).clip(lower=0)
            ).where(lower.notna() & upper.notna())
        result.append(group)
    return pd.concat(result, ignore_index=True)


def metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, horizon_data in predictions.groupby("horizon"):
        pivot = horizon_data.pivot_table(index="target_week_start", columns="model", values="prediction")
        common_targets = pivot.dropna(subset=COMMON_TARGET_MODELS).index
        common = horizon_data[horizon_data["target_week_start"].isin(common_targets)]
        peak_threshold = common.drop_duplicates("target_week_start")["actual"].quantile(0.75)
        for model, group in common.groupby("model"):
            error = group["prediction"] - group["actual"]
            interval = group.dropna(subset=["lower_80_conformal", "upper_80_conformal"])
            asym = group.dropna(subset=["lower_80_asymmetric", "upper_80_asymmetric"])
            rows.append({
                "horizon": int(horizon),
                "model": model,
                "n_common": int(len(group)),
                "mae": float(error.abs().mean()),
                "rmse": float(np.sqrt(np.mean(error ** 2))),
                "wape": float(error.abs().sum() / group["actual"].sum()),
                "mean_bias": float(error.mean()),
                "peak_week_mae": float(error.abs()[group["actual"].ge(peak_threshold)].mean()),
                "coverage_80_asymmetric": float(asym["covered_80_asymmetric"].mean()),
                "coverage_80_conformal": float(interval["covered_80_conformal"].mean()),
                "coverage_80_conformal_2026": float(
                    interval.loc[
                        interval["target_week_start"].astype(str).str.startswith("2026"),
                        "covered_80_conformal",
                    ].mean()
                ),
                "width_80_conformal": float(interval["width_80_conformal"].mean()),
                "interval_score_80_conformal": float(interval["interval_score_80_conformal"].mean()),
            })
    result = pd.DataFrame(rows).sort_values(["horizon", "mae"]).reset_index(drop=True)
    result["mae_rank"] = result.groupby("horizon")["mae"].rank(method="min").astype(int)
    persistence = result[result["model"].eq("last")].set_index("horizon")
    result["mae_skill_vs_persistence"] = result.apply(
        lambda row: 1.0 - row["mae"] / persistence.loc[row["horizon"], "mae"], axis=1
    )
    result["wape_skill_vs_persistence"] = result.apply(
        lambda row: 1.0 - row["wape"] / persistence.loc[row["horizon"], "wape"], axis=1
    )
    return result


def metrics_by_season(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    frame = predictions.copy()
    frame["target_year"] = pd.to_datetime(frame["target_week_start"]).dt.year
    for horizon, horizon_data in frame.groupby("horizon"):
        pivot = horizon_data.pivot_table(
            index="target_week_start", columns="model", values="prediction"
        )
        common_targets = pivot.dropna(subset=COMMON_TARGET_MODELS).index
        common = horizon_data[horizon_data["target_week_start"].isin(common_targets)]
        for target_year, season_data in common.groupby("target_year"):
            unique_actual = season_data.drop_duplicates("target_week_start")[
                ["target_week_start", "actual"]
            ]
            peak_threshold = unique_actual["actual"].quantile(0.75)
            for model, group in season_data.groupby("model"):
                error = group["prediction"] - group["actual"]
                interval = group.dropna(
                    subset=["lower_80_conformal", "upper_80_conformal"]
                )
                rows.append({
                    "horizon": int(horizon),
                    "target_year": int(target_year),
                    "model": model,
                    "n_common": int(len(group)),
                    "mae": float(error.abs().mean()),
                    "rmse": float(np.sqrt(np.mean(error ** 2))),
                    "wape": float(error.abs().sum() / group["actual"].sum()),
                    "mean_bias": float(error.mean()),
                    "peak_week_mae": float(
                        error.abs()[group["actual"].ge(peak_threshold)].mean()
                    ),
                    "coverage_80_conformal": float(
                        interval["covered_80_conformal"].mean()
                    ),
                    "width_80_conformal": float(
                        interval["width_80_conformal"].mean()
                    ),
                    "interval_score_80_conformal": float(
                        interval["interval_score_80_conformal"].mean()
                    ),
                })
    return pd.DataFrame(rows).sort_values(
        ["horizon", "target_year", "mae"]
    ).reset_index(drop=True)


def moving_block_sample(
    values_by_year: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for values in values_by_year.values():
        n = len(values)
        selected: list[np.ndarray] = []
        while len(selected) < n:
            start = int(rng.integers(0, n))
            selected.extend(
                values[(start + offset) % n]
                for offset in range(CONFIG.bootstrap_block_length)
            )
        sampled.extend(selected[:n])
    return np.asarray(sampled)


def paired_forecast_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Paired, year-stratified moving-block intervals for model differences."""
    rng = np.random.default_rng(CONFIG.random_state + 17)
    comparisons = [
        "last", "negative_binomial_glm", "log_growth4",
        "damped_holt_log", "ensemble_log_hgb", "bio_hgb", "bio_constrained", "rolling_selected_hgb",
        "rolling_selected_all",
    ]
    rows: list[dict] = []
    for horizon, horizon_data in predictions.groupby("horizon", sort=True):
        values = horizon_data.pivot_table(
            index="target_week_start", columns="model", values="prediction"
        )
        actual = horizon_data.drop_duplicates("target_week_start").set_index(
            "target_week_start"
        )["actual"]
        for challenger in comparisons:
            paired = pd.DataFrame({
                "actual": actual,
                "reference": values["baseline_hgb"],
                "challenger": values[challenger],
            }).dropna()
            paired["target_year"] = pd.to_datetime(paired.index).year
            paired["absolute_error_difference"] = (
                (paired["challenger"] - paired["actual"]).abs()
                - (paired["reference"] - paired["actual"]).abs()
            )
            paired["squared_error_difference"] = (
                (paired["challenger"] - paired["actual"]) ** 2
                - (paired["reference"] - paired["actual"]) ** 2
            )
            observed_mae = float(paired["absolute_error_difference"].mean())
            observed_wape = float(
                paired["absolute_error_difference"].sum() / paired["actual"].sum()
            )
            arrays = {
                int(year): group[
                    ["absolute_error_difference", "squared_error_difference", "actual"]
                ].to_numpy()
                for year, group in paired.groupby("target_year", sort=True)
            }
            mae_draws = np.empty(CONFIG.bootstrap_replicates)
            wape_draws = np.empty(CONFIG.bootstrap_replicates)
            for replicate in range(CONFIG.bootstrap_replicates):
                draw = moving_block_sample(arrays, rng)
                mae_draws[replicate] = draw[:, 0].mean()
                wape_draws[replicate] = draw[:, 0].sum() / draw[:, 2].sum()
            rows.append({
                "horizon": int(horizon),
                "reference_model": "baseline_hgb",
                "challenger_model": challenger,
                "paired_targets": int(len(paired)),
                "challenger_minus_reference_mae": observed_mae,
                "mae_difference_95pct_ci_lower": float(np.quantile(mae_draws, 0.025)),
                "mae_difference_95pct_ci_upper": float(np.quantile(mae_draws, 0.975)),
                "probability_challenger_lower_mae": float(np.mean(mae_draws < 0.0)),
                "challenger_minus_reference_wape": observed_wape,
                "wape_difference_95pct_ci_lower": float(np.quantile(wape_draws, 0.025)),
                "wape_difference_95pct_ci_upper": float(np.quantile(wape_draws, 0.975)),
                "bootstrap_replicates": CONFIG.bootstrap_replicates,
                "moving_block_length_weeks": CONFIG.bootstrap_block_length,
            })
    return pd.DataFrame(rows)


def selection_table(metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics[metrics["model"].isin(MODEL_SELECTION_CANDIDATES)]
    selected = candidates.sort_values(
        ["horizon", "mae", "interval_score_80_conformal"]
    ).groupby("horizon", as_index=False).first()
    return selected.rename(columns={"model": "selected_model"})


def fit_current(
    df: pd.DataFrame,
    base: pd.DataFrame,
    baseline: list[str],
    biology: list[str],
    directions: dict[str, int],
    predictions: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    eligible = df.index[df["target_complete"]]
    origin = int(eligible.max())
    output = []
    for selected in selection.itertuples():
        horizon = int(selected.horizon)
        frame, baseline_cols, bio_cols = horizon_frame(df, base, baseline, biology, horizon)
        train = frame[
            frame["origin_eligible"]
            & frame["target_complete_future"]
            & frame["target"].notna()
            & frame["row_index"].le(origin - horizon)
        ]
        model_name = selected.selected_model
        if model_name in ML_MODELS:
            columns = baseline_cols if model_name == "baseline_hgb" else bio_cols
            monotonic = None
            if model_name == "bio_constrained":
                monotonic = [directions.get(column, 0) for column in bio_cols]
            model = median_pipeline(monotonic)
            model.fit(train[columns], train["target"].astype(float))
            point = max(0.0, float(model.predict(frame.loc[[origin], columns])[0]))
        elif model_name == "log_growth4":
            point = log_growth_prediction(
                df.loc[origin - 3:origin, "observed_target"], horizon
            )
        elif model_name == "damped_holt_log":
            point = damped_holt_log_prediction(
                df.loc[:origin, "observed_target"], horizon
            )
        elif model_name == "ensemble_log_hgb":
            hgb_model = median_pipeline()
            hgb_model.fit(train[baseline_cols], train["target"].astype(float))
            hgb_point = max(
                0.0, float(hgb_model.predict(frame.loc[[origin], baseline_cols])[0])
            )
            growth_point = log_growth_prediction(
                df.loc[origin - 3:origin, "observed_target"], horizon
            )
            point = 0.5 * (hgb_point + growth_point)
        elif model_name == "last":
            point = float(df.at[origin, "observed_target"])
        elif model_name == "mean4":
            point = float(df.loc[origin - 3:origin, "observed_target"].mean())
        elif model_name == "seasonal52":
            seasonal_index = origin + horizon - 52
            point = float(df.at[seasonal_index, "observed_target"])
        elif model_name == "negative_binomial_glm":
            classical_count_cols = [
                "target_lag_0", "target_lag_1", "target_lag_2", "target_lag_4",
                "target_lag_8", "target_lag_13", "target_lag_52",
                "target_roll_mean_4", "target_roll_mean_13", "target_roll_mean_26",
            ]
            classical_cols = [
                *classical_count_cols, "target_week_sin", "target_week_cos"
            ]
            nb_train = train[classical_cols].copy()
            nb_target = frame.loc[[origin], classical_cols].copy()
            nb_train[classical_count_cols] = np.log1p(
                nb_train[classical_count_cols].clip(lower=0)
            )
            nb_target[classical_count_cols] = np.log1p(
                nb_target[classical_count_cols].clip(lower=0)
            )
            point, _, _, _ = negative_binomial_prediction(
                nb_train, train["target"].astype(float), nb_target,
            )
        else:
            raise ValueError(f"Unsupported selected model for current forecast: {model_name}")

        calibration = predictions[
            predictions["horizon"].eq(horizon)
            & predictions["model"].eq(selected.selected_model)
            & predictions["target_index"].le(origin)
        ].dropna(subset=["residual", "prediction"]).sort_values("origin_index").tail(
            CONFIG.interval_window
        )
        n = len(calibration)
        rank_level = min(1.0, math.ceil((n + 1) * (1 - CONFIG.interval_alpha)) / n)
        scores = calibration["residual"].abs() / np.sqrt(
            calibration["prediction"].clip(lower=1.0)
        )
        radius = (
            float(scores.quantile(rank_level, interpolation="higher"))
            * math.sqrt(max(point, 1.0))
            * CONFIG.interval_safety_factor
        )
        target_start = df.at[origin, "week_start"] + pd.Timedelta(weeks=horizon)
        output.append({
            "forecast_origin_week_start": df.at[origin, "week_start"].date().isoformat(),
            "forecast_origin_week_end": df.at[origin, "week_end"].date().isoformat(),
            "horizon_weeks": horizon,
            "target_week_start": target_start.date().isoformat(),
            "target_week_end": (target_start + pd.Timedelta(days=6)).date().isoformat(),
            "selected_model": selected.selected_model,
            "point_forecast_admissions": round(point, 1),
            "lower_80_conformal": round(max(0.0, point - radius), 1),
            "upper_80_conformal": round(point + radius, 1),
            "calibration_forecasts": n,
            "interval_calibration_window_weeks": CONFIG.interval_window,
            "interval_safety_factor": CONFIG.interval_safety_factor,
            "backtest_wape": selected.wape,
            "backtest_coverage_80_conformal": selected.coverage_80_conformal,
            "backtest_coverage_80_conformal_2026": selected.coverage_80_conformal_2026,
            "weather_at_origin_complete": bool(df.at[origin, "weather_complete"]),
            "status": "research forecast; not an operational allocation limit",
        })
    return pd.DataFrame(output)


def write_report(metrics: pd.DataFrame, selection: pd.DataFrame, current: pd.DataFrame) -> str:
    comparison = metrics[metrics["model"].isin(ML_MODELS)].copy()
    lines = []
    for row in selection.itertuples():
        baseline = comparison[
            comparison["horizon"].eq(row.horizon) & comparison["model"].eq("baseline_hgb")
        ].iloc[0]
        improvement = (baseline.mae - row.mae) / baseline.mae
        lines.append(
            f"- {row.horizon}-week: **{row.selected_model}**, MAE {row.mae:,.1f}, "
            f"WAPE {row.wape:.1%}, MAE improvement versus baseline {improvement:.1%}, "
            f"planning-interval coverage {row.coverage_80_conformal:.1%} overall and "
            f"{row.coverage_80_conformal_2026:.1%} in the 2026 evaluation period."
        )
    forecast_lines = [
        f"- Week beginning {row.target_week_start}: {row.point_forecast_admissions:,.0f} "
        f"admissions (80% conformal interval {row.lower_80_conformal:,.0f}-"
        f"{row.upper_80_conformal:,.0f})."
        for row in current.itertuples()
    ]
    return f"""# Biology-informed weekly dengue forecast benchmark

Generated: {datetime.now().astimezone().isoformat()}

## Result

{chr(10).join(lines)}

The comparison used identical expanding-window forecast origins, target weeks,
and predictor missingness rules. A biology-informed model is not retained merely
because it is more interpretable; it must improve out-of-sample performance.

## Biology encoded

- Nonnegative counts through a Poisson loss.
- A transparent Ae. aegypti thermal-suitability feature that is zero at 17.8 C
  and 34.6 C and peaks at 29.1 C, following the posterior-mean limits reported
  by Mordecai et al. (2017).
- Distributed weather lags from zero to twelve weeks.
- Heat-stress, heavy-rain, humidity, and temperature-rain interaction features.
- In the constrained sensitivity model, forecasts are monotonically increasing
  in thermal suitability and decreasing in heat stress, holding all other
  predictors fixed.

Raw temperature and rainfall are not forced to have globally positive effects.
Suitability is not sufficient for case growth, and heavy rainfall can flush
immature mosquitoes. The triangular curve is explicitly a sensitivity
approximation, not a reproduction of the original mechanistic R0(T) equation.

Scientific references:

- https://doi.org/10.1371/journal.pntd.0005568
- https://doi.org/10.1371/journal.pntd.0007213
- https://doi.org/10.1371/journal.pcbi.1013540

## Recalibrated current research forecasts

{chr(10).join(forecast_lines)}

Intervals use sequential, locally scaled conformal calibration. At each
historical origin, only forecast errors whose target week had already resolved
were eligible. Absolute errors are divided by the square root of the point
forecast (a count-data variance stabilizer), the latest 52 eligible errors are
used, and the restored interval radius receives a transparent 10% planning
safety expansion. The nominal target is 80%; time-series dependence means this
is an empirical planning interval, not an exchangeability-based coverage
guarantee.

## Interpretation limit

This is biology-informed feature engineering with a constrained sensitivity
model, not a fully identified human-mosquito SEI-SEIR model. A defensible full
hybrid requires serotype/immunity, mosquito abundance, intervention, and
infection-to-admission observation data that are not currently available.
"""


def write_checkpoint(manifest: dict, selection: pd.DataFrame) -> None:
    text = f"""# Biology-informed forecast checkpoint

Last updated: {datetime.now().astimezone().isoformat()}

## Status

- Verified weekly target overlay: complete.
- Biology feature construction: complete.
- Rolling-origin 1-, 2-, and 4-week comparison: complete.
- Sequential 52-week, count-scaled conformal recalibration: complete.
- Status: {manifest['status']}.

## Resume

Run `python3 work/biology_forecast/build_biology_benchmark.py`. Completed horizon
checkpoints are reused only when the verified target, daily weather, and
configuration hashes match.

## Selected models

{selection.to_string(index=False)}
"""
    (WORK / "CHECKPOINT.md").write_text(text, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    manifest = initialize_manifest()
    df = load_data()
    base, baseline, biology, directions = build_features(df)
    df[["week_start", *BIO_COLUMNS, "bio_weather_complete"]].to_csv(
        RESULTS / "biology_features_weekly.csv", index=False
    )
    manifest["status"] = "features_complete"
    if "biology_features" not in manifest["completed_stages"]:
        manifest["completed_stages"].append("biology_features")
    save_manifest(manifest)

    horizon_outputs = []
    for horizon in CONFIG.horizons:
        checkpoint = RESULTS / f"predictions_h{horizon}.csv"
        if horizon in manifest.get("completed_horizons", []) and checkpoint.exists():
            horizon_outputs.append(pd.read_csv(checkpoint))
            continue
        predictions = predict_horizon(df, base, baseline, biology, directions, horizon)
        predictions.to_csv(checkpoint, index=False)
        horizon_outputs.append(predictions)
        if horizon not in manifest["completed_horizons"]:
            manifest["completed_horizons"].append(horizon)
        manifest["status"] = f"horizon_{horizon}_complete"
        save_manifest(manifest)

    predictions = add_rolling_model_selection(
        pd.concat(horizon_outputs, ignore_index=True)
    )
    predictions = add_intervals(predictions)
    metrics = metrics_table(predictions)
    seasonal_metrics = metrics_by_season(predictions)
    bootstrap = paired_forecast_bootstrap(predictions)
    selection = selection_table(metrics)
    current = fit_current(df, base, baseline, biology, directions, predictions, selection)
    report = write_report(metrics, selection, current)

    files = {
        "biology_backtest_predictions.csv": predictions,
        "biology_benchmark_metrics.csv": metrics,
        "biology_benchmark_metrics_by_season.csv": seasonal_metrics,
        "biology_paired_bootstrap.csv": bootstrap,
        "biology_selected_models.csv": selection,
        "biology_current_forecast.csv": current,
    }
    for name, data in files.items():
        data.to_csv(RESULTS / name, index=False)
        data.to_csv(OUTPUTS / f"dengue_{name}", index=False)
    (RESULTS / "biology_benchmark_report.md").write_text(report, encoding="utf-8")
    (OUTPUTS / "dengue_biology_benchmark_report.md").write_text(report, encoding="utf-8")
    shutil.copy2(RESULTS / "biology_features_weekly.csv", OUTPUTS / "dengue_biology_features_weekly.csv")
    shutil.copy2(Path(__file__), OUTPUTS / "dengue_biology_forecast_pipeline.py")

    manifest["status"] = "complete"
    for stage in [
        "rolling_backtest", "rolling_model_selection", "conformal_intervals",
        "seasonal_metrics", "paired_bootstrap", "metrics", "deliverables",
    ]:
        if stage not in manifest["completed_stages"]:
            manifest["completed_stages"].append(stage)
    manifest["selected_models"] = selection.to_dict(orient="records")
    save_manifest(manifest)
    write_checkpoint(manifest, selection)
    shutil.copy2(WORK / "CHECKPOINT.md", OUTPUTS / "dengue_biology_resume_checkpoint.md")
    print(selection.to_string(index=False))
    print(json.dumps({"status": "complete", "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
