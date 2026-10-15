#!/usr/bin/env python3
"""Quick, offline integrity and headline-result checks for the release."""

from __future__ import annotations

import csv
import hashlib
import math
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {relative}")
    return path


def rows(relative: str) -> list[dict[str, str]]:
    with require(relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def dghs_week(day: date) -> tuple[int, int, date, date]:
    week_year = day.year
    jan1 = date(week_year, 1, 1)
    start1 = jan1 - timedelta(days=(jan1.weekday() + 1) % 7)
    if day < start1:
        week_year -= 1
        jan1 = date(week_year, 1, 1)
        start1 = jan1 - timedelta(days=(jan1.weekday() + 1) % 7)
    number = (day - start1).days // 7 + 1
    start = start1 + timedelta(weeks=number - 1)
    return week_year, number, start, start + timedelta(days=6)


def verify_headlines() -> None:
    year, number, start, end = dghs_week(date(2026, 8, 9))
    if (year, number, start, end) != (
        2026, 33, date(2026, 8, 9), date(2026, 8, 15)
    ):
        raise AssertionError("DGHS calendar validation failed")

    metrics = rows("outputs/dengue_biology_benchmark_metrics.csv")
    forecast = next(
        row for row in metrics
        if row["horizon"] == "1" and row["model"] == "ensemble_log_hgb"
    )
    if int(float(forecast["n_common"])) != 124:
        raise AssertionError("Unexpected one-week target count")
    close(float(forecast["mae"]), 226.4933, 0.01, "one-week MAE")
    close(float(forecast["wape"]), 0.098342, 0.00001, "one-week WAPE")
    close(
        float(forecast["coverage_80_conformal"]), 0.855769, 0.00001,
        "one-week interval coverage",
    )

    allocation_rows = rows("outputs/dengue_phase2_joint_backtest_summary.csv")
    allocation = next(
        row for row in allocation_rows
        if row["analysis_rule"] == "published_primary"
        and row["alignment_scenario"] == "confirmed_dghs"
        and row["forecast_level"] == "point"
        and row["policy"] == "pooled3"
    )
    if int(float(allocation["target_weeks"])) != 101:
        raise AssertionError("Unexpected allocation target count")
    close(float(allocation["mean_shortage_fraction"]), 0.145248, 0.00001, "mean shortage")
    close(
        float(allocation["demand_weighted_shortage_fraction"]),
        0.111026, 0.00001, "weighted shortage",
    )
    close(
        float(allocation["aggregate_budget_to_actual_ratio"]),
        0.967616, 0.00001, "aggregate budget ratio",
    )


def verify_manifest() -> int:
    manifest = ROOT / "MANIFEST.sha256"
    if not manifest.exists():
        return 0
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = require(relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    required = [
        "README.md", "DATA_AVAILABILITY.md", "CODEBOOK.md", "CITATION.cff",
        "manuscript/main.tex", "manuscript/references.bib",
        "data/processed/dengue_weekly_model_verified.csv",
        "data/processed/dengue_weekly_division_published.csv",
        "outputs/dengue_biology_benchmark_metrics.csv",
        "outputs/dengue_phase2_joint_backtest_summary.csv",
    ]
    for relative in required:
        require(relative)
    verify_headlines()
    checked = verify_manifest()
    suffix = f" and {checked} checksums" if checked else ""
    print(f"Repository verification passed{suffix}.")


if __name__ == "__main__":
    main()
