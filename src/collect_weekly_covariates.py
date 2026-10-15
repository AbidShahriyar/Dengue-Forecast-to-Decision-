#!/usr/bin/env python3
"""Collect provisional Bangladesh weather covariates and build weekly model tables.

Weather is downloaded from NASA POWER at the eight divisional-capital points.
These points are transparent, reproducible proxies, not polygon averages. The
script aggregates both weather and DGHS daily observations to the DGHS
Sunday-Saturday reporting week. Week 1 is the Sunday-Saturday interval that
contains 1 January; for example, 9-15 August 2026 is week 33.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_DOCS = "https://power.larc.nasa.gov/docs/services/api/temporal/daily/"
PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX", "RH2M"]
FILL_VALUE = -999.0

# Divisional-capital coordinates. These are provisional point proxies for the
# surrounding divisions and must not be described as division-wide averages.
LOCATIONS = [
    {"division": "Barishal", "proxy_city": "Barishal", "latitude": 22.7010, "longitude": 90.3535},
    {"division": "Chattogram", "proxy_city": "Chattogram", "latitude": 22.3569, "longitude": 91.7832},
    {"division": "Dhaka", "proxy_city": "Dhaka", "latitude": 23.8103, "longitude": 90.4125},
    {"division": "Khulna", "proxy_city": "Khulna", "latitude": 22.8456, "longitude": 89.5403},
    {"division": "Mymensingh", "proxy_city": "Mymensingh", "latitude": 24.7471, "longitude": 90.4203},
    {"division": "Rajshahi", "proxy_city": "Rajshahi", "latitude": 24.3745, "longitude": 88.6042},
    {"division": "Rangpur", "proxy_city": "Rangpur", "latitude": 25.7439, "longitude": 89.2752},
    {"division": "Sylhet", "proxy_city": "Sylhet", "latitude": 24.8949, "longitude": 91.8687},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="20180101", help="NASA start date YYYYMMDD")
    parser.add_argument("--end", default="20261011", help="NASA end date YYYYMMDD")
    parser.add_argument(
        "--dghs-dir",
        type=Path,
        default=Path("outputs/dghs_sample"),
        help="Directory containing dashboard_<year>.json files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work/weekly_covariates"),
        help="Output directory",
    )
    return parser.parse_args()


def request_json(url: str, attempts: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # network/API failures are retried conservatively
            last_error = error
            if attempt < attempts:
                time.sleep(attempt * 2)
    raise RuntimeError(f"NASA POWER request failed after {attempts} attempts: {last_error}")


def valid_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or math.isclose(float(value), FILL_VALUE):
        return None
    return float(value)


def dghs_week_fields(day: date) -> tuple[int, int, date, date]:
    """Return DGHS week-year, number, Sunday start, and Saturday end.

    The week that contains 1 January is week 1. Consequently, week 1 may begin
    in the preceding calendar year. This definition reproduces the confirmed
    example that 9-15 August 2026 is week 33.
    """
    start = day - timedelta(days=(day.weekday() + 1) % 7)
    end = start + timedelta(days=6)
    week_year = end.year if end.year != start.year else start.year
    anchor = date(week_year, 1, 1)
    first_start = anchor - timedelta(days=(anchor.weekday() + 1) % 7)
    week_number = ((start - first_start).days // 7) + 1
    return week_year, week_number, start, end


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_weather(start: str, end: str, out_dir: Path) -> list[dict[str, Any]]:
    raw_dir = out_dir / "nasa_power_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for location in LOCATIONS:
        query = urllib.parse.urlencode(
            {
                "parameters": ",".join(PARAMETERS),
                "community": "AG",
                "longitude": location["longitude"],
                "latitude": location["latitude"],
                "start": start,
                "end": end,
                "format": "JSON",
                "time-standard": "UTC",
            }
        )
        url = f"{POWER_URL}?{query}"
        raw_path = raw_dir / f"{location['division'].lower()}_{start}_{end}.json"
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            payload = request_json(url)
            raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        parameter = payload["properties"]["parameter"]
        date_keys = sorted(set().union(*(series.keys() for series in parameter.values())))
        for key in date_keys:
            day = datetime.strptime(key, "%Y%m%d").date()
            week_year, week_number, week_start, week_end = dghs_week_fields(day)
            rows.append(
                {
                    "division": location["division"],
                    "proxy_city": location["proxy_city"],
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "date": day.isoformat(),
                    "week_year": week_year,
                    "week_number": week_number,
                    "week_start": week_start.isoformat(),
                    "week_end": week_end.isoformat(),
                    "precip_mm": valid_number(parameter["PRECTOTCORR"].get(key)),
                    "temp_mean_c": valid_number(parameter["T2M"].get(key)),
                    "temp_min_c": valid_number(parameter["T2M_MIN"].get(key)),
                    "temp_max_c": valid_number(parameter["T2M_MAX"].get(key)),
                    "rh_mean_pct": valid_number(parameter["RH2M"].get(key)),
                    "source_url": url,
                }
            )
    return rows


def numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def aggregate_weather_weekly(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        groups[(row["division"], row["week_year"], row["week_number"])].append(row)

    result: list[dict[str, Any]] = []
    for (division, week_year, week_number), group in sorted(groups.items()):
        precipitation = numeric(row["precip_mm"] for row in group)
        temperature = numeric(row["temp_mean_c"] for row in group)
        minimums = numeric(row["temp_min_c"] for row in group)
        maximums = numeric(row["temp_max_c"] for row in group)
        humidity = numeric(row["rh_mean_pct"] for row in group)
        first = group[0]
        result.append(
            {
                "division": division,
                "proxy_city": first["proxy_city"],
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "week_year": week_year,
                "week_number": week_number,
                "week_start": min(row["week_start"] for row in group),
                "week_end": max(row["week_end"] for row in group),
                "calendar_rows": len(group),
                "valid_days_precip": len(precipitation),
                "valid_days_temp": len(temperature),
                "valid_days_rh": len(humidity),
                "weather_complete": (
                    len(group) == 7
                    and len(precipitation) == 7
                    and len(temperature) == 7
                    and len(humidity) == 7
                ),
                "precip_sum_mm": round(sum(precipitation), 3) if precipitation else None,
                "precip_mean_mm_day": round(mean(precipitation), 3) if precipitation else None,
                "rainy_days_ge_1mm": sum(1 for value in precipitation if value >= 1.0),
                "temp_mean_c": round(mean(temperature), 3) if temperature else None,
                "temp_weekly_min_c": round(min(minimums), 3) if minimums else None,
                "temp_weekly_max_c": round(max(maximums), 3) if maximums else None,
                "rh_mean_pct": round(mean(humidity), 3) if humidity else None,
                "source": "NASA POWER daily point data; divisional-capital proxy",
                "source_url": POWER_DOCS,
            }
        )
    return result


def aggregate_weather_national(weekly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in weekly_rows:
        groups[(row["week_year"], row["week_number"])].append(row)
    result: list[dict[str, Any]] = []
    average_fields = [
        "precip_sum_mm",
        "precip_mean_mm_day",
        "rainy_days_ge_1mm",
        "temp_mean_c",
        "temp_weekly_min_c",
        "temp_weekly_max_c",
        "rh_mean_pct",
    ]
    for (week_year, week_number), group in sorted(groups.items()):
        row: dict[str, Any] = {
            "week_year": week_year,
            "week_number": week_number,
            "week_start": min(item["week_start"] for item in group),
            "week_end": max(item["week_end"] for item in group),
            "division_proxies_observed": len(group),
            "division_proxies_complete": sum(1 for item in group if item["weather_complete"]),
            "weather_complete": len(group) == 8 and all(item["weather_complete"] for item in group),
        }
        for field in average_fields:
            values = numeric(item[field] for item in group)
            row[f"proxy_mean_{field}"] = round(mean(values), 3) if values else None
        row["source"] = "Unweighted mean of eight NASA POWER divisional-capital point proxies"
        row["source_url"] = POWER_DOCS
        result.append(row)
    return result


def chart(snapshot: dict[str, Any], chart_id: str) -> dict[str, Any] | None:
    return next((item for item in snapshot["charts"] if item["chart_id"] == chart_id), None)


def aggregate_dengue_weekly(dghs_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    national: list[dict[str, Any]] = []
    published_division: list[dict[str, Any]] = []

    for path in sorted(dghs_dir.glob("dashboard_*.json")):
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        selected_year = int(snapshot["selected_year"])
        admissions_chart = chart(snapshot, "confirmed_case")
        deaths_chart = chart(snapshot, "death_case")
        if not admissions_chart or not deaths_chart:
            continue

        weekly: dict[tuple[int, int, date, date], dict[str, Any]] = defaultdict(
            lambda: {"admissions": 0, "admission_days": 0, "deaths": 0, "death_days_reported": 0}
        )
        invalid_admission_rows = 0
        for point in admissions_chart["series"][0]["points"]:
            try:
                day = datetime.strptime(point["category"], "%d-%b-%y").date()
            except ValueError:
                invalid_admission_rows += 1
                continue
            if day.year != selected_year:
                invalid_admission_rows += 1
                continue
            week_year, week_number, week_start, week_end = dghs_week_fields(day)
            key = (week_year, week_number, week_start, week_end)
            weekly[key]["admissions"] += point["y"] or 0
            weekly[key]["admission_days"] += 1

        for point in deaths_chart["series"][0]["points"]:
            try:
                day = datetime.strptime(point["category"], "%d-%b-%y").date()
            except ValueError:
                continue
            if day.year != selected_year:
                continue
            week_year, week_number, week_start, week_end = dghs_week_fields(day)
            key = (week_year, week_number, week_start, week_end)
            weekly[key]["deaths"] += point["y"] or 0
            weekly[key]["death_days_reported"] += 1

        for (week_year, week_number, week_start, week_end), values in sorted(weekly.items()):
            national.append(
                {
                    "selected_year": selected_year,
                    "week_year": week_year,
                    "week_number": week_number,
                    "week_start": week_start.isoformat(),
                    "week_end": week_end.isoformat(),
                    "admissions": values["admissions"],
                    "admission_days_published": values["admission_days"],
                    "deaths": values["deaths"],
                    "death_days_with_nonzero_reports": values["death_days_reported"],
                    "invalid_or_out_of_year_admission_rows_in_source": invalid_admission_rows,
                    "source": "DGHS daily dashboard series aggregated to Sunday-Saturday weeks",
                    "source_url": snapshot["source_url"],
                    "quality_note": "Missing daily admission dates are not imputed; absent daily death dates are treated as zero in the weekly sum.",
                }
            )

        division_chart = chart(snapshot, "affected_in_division_by_week")
        if division_chart:
            for series in division_chart["series"]:
                for point in series["points"]:
                    published_division.append(
                        {
                            "selected_year": selected_year,
                            "division_label": series["name"],
                            "published_week_label": point["category"],
                            "admissions": point["y"],
                            "source_chart_id": "affected_in_division_by_week",
                            "source_url": snapshot["source_url"],
                            "quality_note": "DGHS Wxx label mapped to the confirmed Sunday-Saturday reporting calendar.",
                        }
                    )
    return national, published_division


def build_model_frame(
    dengue_weekly: list[dict[str, Any]], weather_national: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    weather_by_week = {
        (row["week_year"], row["week_number"]): row for row in weather_national
    }
    result: list[dict[str, Any]] = []
    for dengue in dengue_weekly:
        weather = weather_by_week.get((dengue["week_year"], dengue["week_number"]))
        row = dict(dengue)
        if weather:
            for key, value in weather.items():
                if key not in {"week_year", "week_number", "week_start", "week_end", "source", "source_url"}:
                    row[key] = value
            row["weather_source_url"] = weather["source_url"]
            row["weather_method"] = weather["source"]
        result.append(row)
    return result


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    daily_weather = fetch_weather(args.start, args.end, args.out)
    weekly_weather = aggregate_weather_weekly(daily_weather)
    national_weather = aggregate_weather_national(weekly_weather)
    dengue_weekly, dengue_division_published = aggregate_dengue_weekly(args.dghs_dir)
    model_frame = build_model_frame(dengue_weekly, national_weather)

    write_csv(args.out / "weather_daily_proxy.csv", daily_weather, list(daily_weather[0]))
    write_csv(args.out / "weather_weekly_proxy.csv", weekly_weather, list(weekly_weather[0]))
    write_csv(args.out / "weather_weekly_national_proxy.csv", national_weather, list(national_weather[0]))
    write_csv(args.out / "dengue_weekly_national.csv", dengue_weekly, list(dengue_weekly[0]))
    write_csv(
        args.out / "dengue_weekly_division_published.csv",
        dengue_division_published,
        list(dengue_division_published[0]),
    )
    write_csv(args.out / "weekly_model_frame.csv", model_frame, list(model_frame[0]))

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "nasa_start_requested": args.start,
        "nasa_end_requested": args.end,
        "weather_parameters": PARAMETERS,
        "location_method": "Eight divisional-capital point proxies",
        "week_definition": "DGHS Sunday-Saturday; week 1 contains 1 January",
        "week_definition_example": "2026 W33 = 2026-08-09 through 2026-08-15",
        "rows": {
            "weather_daily_proxy": len(daily_weather),
            "weather_weekly_proxy": len(weekly_weather),
            "weather_weekly_national_proxy": len(national_weather),
            "dengue_weekly_national": len(dengue_weekly),
            "dengue_weekly_division_published": len(dengue_division_published),
            "weekly_model_frame": len(model_frame),
        },
        "sources": {
            "weather": POWER_DOCS,
            "dengue": "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
