#!/usr/bin/env python3
"""Build an auditable pre-2024 Bangladesh dengue history and weekly model frame.

The historical daily/monthly source is pinned to the data commit cited by:
  Bhowmik et al. (2026), Tropical Medicine & International Health.
  https://doi.org/10.1111/tmi.70136

OpenDengue v1.3 is retained as an independent comparison source:
  https://doi.org/10.6084/m9.figshare.24259573
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


PINNED_COMMIT = "c67c4dac43ed9d00bb37dbbb15c1da91b730f6b5"
RAW_BASE = (
    "https://raw.githubusercontent.com/awnonbhowmik/"
    f"DENV-Data-Analysis/{PINNED_COMMIT}/data"
)
FILES = {
    "daily": f"{RAW_BASE}/Daily_Data_2019-2024.xlsx",
    "monthly_cases": f"{RAW_BASE}/Dengue_Monthly_Infection_Cleaned_2001-2024.xlsx",
    "monthly_deaths": f"{RAW_BASE}/Monthly_Death_2019-2024.xlsx",
    "regional": f"{RAW_BASE}/DENV_Regional_2019_2024.xlsx",
    "opendengue_temporal": (
        "https://github.com/OpenDengue/master-repo/raw/main/data/releases/"
        "V1.3/Temporal_extract_V1_3.zip"
    ),
}

PAPER_DOI = "https://doi.org/10.1111/tmi.70136"
PAPER_DATA_URL = (
    "https://github.com/awnonbhowmik/DENV-Data-Analysis/tree/"
    f"{PINNED_COMMIT}/data"
)
OPENDENGUE_DOI = "https://doi.org/10.6084/m9.figshare.24259573"
WHO_2023_DON = "https://www.who.int/emergencies/disease-outbreak-news/item/2023-DON481"
WHO_2022_DON = "https://www.who.int/emergencies/disease-outbreak-news/item/2022-DON424"
WHO_2023_SITREP = (
    "https://cdn.who.int/media/docs/default-source/searo/bangladesh/"
    "dengue-sitrep/dengue-sitrep-issue-12.pdf"
)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def filter_opendengue(source: Path, output: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(source) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                if row.get("ISO_A0") == "BGD" or row.get("adm_0_name") == "BANGLADESH":
                    rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def reshape_daily(path: Path) -> pd.DataFrame:
    wide = pd.read_excel(path, sheet_name="Daily")
    blocks = []
    for offset in range(0, len(wide.columns), 2):
        date_col = wide.columns[offset]
        value_col = wide.columns[offset + 1]
        block = wide[[date_col, value_col]].copy()
        block.columns = ["date", "admissions"]
        block["date"] = pd.to_datetime(block["date"], errors="coerce")
        block["admissions"] = pd.to_numeric(block["admissions"], errors="coerce")
        block = block.dropna(subset=["date", "admissions"])
        block["admissions"] = block["admissions"].astype(int)
        block["source_column_pair"] = f"{date_col}/{value_col}"
        blocks.append(block)
    daily = pd.concat(blocks, ignore_index=True).drop_duplicates(subset=["date"], keep="first")
    daily = daily[daily["date"].dt.year < 2024].sort_values("date").reset_index(drop=True)
    daily["calendar_year"] = daily["date"].dt.year
    daily["source"] = "Bhowmik et al. pinned daily data"
    daily["source_url"] = PAPER_DATA_URL
    daily["citation_doi"] = PAPER_DOI
    daily["coverage_scope"] = daily["calendar_year"].map(
        {
            2019: "partial: starts 2019-07-29",
            2020: "partial: ends 2020-01-30",
            2021: "partial: starts 2021-12-13",
            2022: "full calendar year",
            2023: "full calendar year",
        }
    )
    return daily


def dghs_week_fields(day: date) -> tuple[int, int, date, date]:
    """Return DGHS Sunday-Saturday week fields; week 1 contains 1 January."""
    start = day - timedelta(days=(day.weekday() + 1) % 7)
    end = start + timedelta(days=6)
    week_year = end.year if end.year != start.year else start.year
    anchor = date(week_year, 1, 1)
    first_start = anchor - timedelta(days=(anchor.weekday() + 1) % 7)
    week_number = ((start - first_start).days // 7) + 1
    return week_year, week_number, start, end


def aggregate_current_weekly(path: Path) -> pd.DataFrame:
    current = pd.read_csv(path)
    current["week_start"] = pd.to_datetime(current["week_start"])
    current["week_end"] = pd.to_datetime(current["week_end"])
    grouped = current.groupby(
        ["week_year", "week_number", "week_start", "week_end"], as_index=False
    ).agg(
        admissions=("admissions", "sum"),
        target_days_observed=("admission_days_published", "sum"),
        deaths=("deaths", "sum"),
        death_days_with_nonzero_reports=("death_days_with_nonzero_reports", "sum"),
        invalid_target_rows=("invalid_or_out_of_year_admission_rows_in_source", "sum"),
    )
    grouped["target_source"] = "DGHS HEOC dengue dashboard"
    grouped["target_source_url"] = "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php"
    grouped["citation_doi"] = ""
    grouped["coverage_scope"] = "dashboard daily series aggregated to DGHS Sunday-Saturday weeks"
    grouped["death_temporal_resolution"] = "daily aggregated to DGHS Sunday-Saturday week"
    return grouped


def aggregate_historical_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    history = daily.copy()
    fields = history["date"].dt.date.map(dghs_week_fields)
    history[["week_year", "week_number", "week_start", "week_end"]] = pd.DataFrame(
        fields.tolist(), index=history.index
    )
    history["week_start"] = pd.to_datetime(history["week_start"])
    history["week_end"] = pd.to_datetime(history["week_end"])
    grouped = history.groupby(
        ["week_year", "week_number", "week_start", "week_end"], as_index=False
    ).agg(
        admissions=("admissions", "sum"),
        target_days_observed=("date", "nunique"),
        coverage_scope=("coverage_scope", lambda values: " | ".join(sorted(set(values.dropna())))),
    )
    grouped["deaths"] = pd.NA
    grouped["death_days_with_nonzero_reports"] = pd.NA
    grouped["invalid_target_rows"] = 0
    grouped["target_source"] = "Bhowmik et al. pinned daily data"
    grouped["target_source_url"] = PAPER_DATA_URL
    grouped["citation_doi"] = PAPER_DOI
    grouped["death_temporal_resolution"] = "not available weekly; monthly deaths stored separately"
    return grouped


def build_weekly_frame(history: pd.DataFrame, current: pd.DataFrame, weather_path: Path) -> pd.DataFrame:
    target = pd.concat([history, current], ignore_index=True).sort_values("week_start")
    target = target.groupby(
        ["week_year", "week_number", "week_start", "week_end"], as_index=False
    ).agg(
        admissions=("admissions", "sum"),
        target_days_observed=("target_days_observed", "sum"),
        deaths=("deaths", lambda values: values.sum(min_count=1)),
        death_days_with_nonzero_reports=(
            "death_days_with_nonzero_reports", lambda values: values.sum(min_count=1)
        ),
        invalid_target_rows=("invalid_target_rows", "sum"),
        target_source=("target_source", lambda values: " | ".join(sorted(set(values.dropna())))),
        target_source_url=(
            "target_source_url", lambda values: " | ".join(sorted(set(values.dropna())))
        ),
        citation_doi=("citation_doi", lambda values: " | ".join(sorted(set(values.dropna())))),
        coverage_scope=("coverage_scope", lambda values: " | ".join(sorted(set(values.dropna())))),
        death_temporal_resolution=(
            "death_temporal_resolution", lambda values: " | ".join(sorted(set(values.dropna())))
        ),
    )
    first_week = target["week_start"].min()
    last_week = target["week_start"].max()
    grid = pd.DataFrame({"week_start": pd.date_range(first_week, last_week, freq="W-SUN")})
    grid["week_end"] = grid["week_start"] + pd.Timedelta(days=6)
    fields = grid["week_start"].dt.date.map(dghs_week_fields)
    grid[["week_year", "week_number", "derived_start", "derived_end"]] = pd.DataFrame(
        fields.tolist(), index=grid.index
    )
    grid = grid.drop(columns=["derived_start", "derived_end"])
    target_payload = target.drop(columns=["week_year", "week_number", "week_end"])
    frame = grid.merge(target_payload, on="week_start", how="left")

    weather = pd.read_csv(weather_path)
    weather["week_start"] = pd.to_datetime(weather["week_start"])
    weather["week_end"] = pd.to_datetime(weather["week_end"])
    weather = weather.drop(columns=["week_year", "week_number", "week_end"])
    frame = frame.merge(weather, on="week_start", how="left")

    frame["target_complete"] = frame["target_days_observed"].eq(7)
    frame["target_available"] = frame["admissions"].notna()
    frame["weather_complete"] = frame["weather_complete"].fillna(False).astype(bool)
    frame["usable_target_weather"] = frame["target_complete"] & frame["weather_complete"]
    complete = frame["target_complete"].fillna(False).astype(int)
    frame["prior_4_target_weeks_complete"] = (
        complete.shift(1).rolling(4, min_periods=4).sum().eq(4)
    )
    frame["usable_for_lagged_model"] = (
        frame["usable_target_weather"] & frame["prior_4_target_weeks_complete"]
    )
    frame["target_gap_reason"] = ""
    frame.loc[~frame["target_available"], "target_gap_reason"] = "no public daily target recovered"
    frame.loc[
        frame["target_available"] & ~frame["target_complete"], "target_gap_reason"
    ] = "fewer than seven daily target observations"
    return frame


def annual_reconciliation(monthly: pd.DataFrame, opendengue: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["year"] = monthly["YearMonth"].astype(str).str[:4].astype(int)
    research = monthly.groupby("year", as_index=False)["Infected"].sum().rename(
        columns={"Infected": "peer_reviewed_monthly_total"}
    )

    od = opendengue.copy()
    od["Year"] = pd.to_numeric(od["Year"], errors="coerce")
    od["dengue_total"] = pd.to_numeric(od["dengue_total"], errors="coerce")
    od = od[(od["T_res"] == "Month") & od["Year"].between(2018, 2024)]
    od_year = od.groupby("Year", as_index=False)["dengue_total"].sum().rename(
        columns={"Year": "year", "dengue_total": "opendengue_monthly_total"}
    )

    official = pd.DataFrame(
        [
            [2018, 10148, 26, WHO_2023_DON, "DGHS totals reproduced by WHO"],
            [2019, 101354, 164, WHO_2023_DON, "DGHS totals reproduced by WHO"],
            [2020, 1405, pd.NA, WHO_2023_SITREP, "WHO notes COVID-era data are limited"],
            [2021, 28429, 105, WHO_2023_DON, "DGHS totals reproduced by WHO"],
            [2022, 62382, 281, WHO_2023_SITREP, "final annual total"],
            [2023, 321179, 1705, WHO_2023_SITREP, "final annual total"],
            [2024, 101214, 575, "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php", "DGHS dashboard annual total"],
        ],
        columns=["year", "official_cases", "official_deaths", "official_source_url", "official_note"],
    )
    result = official.merge(research, on="year", how="left").merge(od_year, on="year", how="left")
    result["peer_minus_official"] = result["peer_reviewed_monthly_total"] - result["official_cases"]
    result["opendengue_minus_official"] = result["opendengue_monthly_total"] - result["official_cases"]
    result["weekly_target_decision"] = result["year"].map(
        {
            2018: "monthly context only",
            2019: "partial daily segment only; do not infer missing weeks",
            2020: "partial daily segment only; reporting disruption",
            2021: "partial daily segment only; do not infer missing weeks",
            2022: "use full daily series",
            2023: "use full daily series",
            2024: "use DGHS dashboard daily series",
        }
    )
    return result


def source_registry() -> pd.DataFrame:
    rows = [
        ["DGHS HEOC dengue dashboard", "Government primary", "Daily/weekly/monthly", "2018 annual; 2024 onward daily in tested filters", "National and division", "Collected", "Primary for 2024 onward", "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php", "Public dashboard blocks plain curl; browser retrieval required; hospitalized/admitted cases"],
        ["Bhowmik et al. data repository", "Peer-reviewed pinned repository", "Daily and monthly", "Daily file 2019-2024; monthly 2001-2024", "National; annual division table", "Collected", "Primary bridge for full daily 2022-2023; partial 2019-2021 retained with flags", PAPER_DATA_URL, f"Paper DOI {PAPER_DOI}; 2020 monthly total differs from WHO"],
        ["DGHS archived dengue press releases", "Government primary", "Daily PDF reports", "Archive visibly includes 2019, 2021, 2022 and 2023 reports", "National and some geographic tables", "Identified; bulk retrieval unavailable", "Primary validation/reconstruction candidate", "https://old.dghs.gov.bd/index.php/bd/home/5200-daily-dengue-status-report", "Archive timed out during programmatic retrieval; no protection bypass attempted"],
        ["WHO Bangladesh dengue situation reports", "Official secondary using DGHS data", "Weekly reports and annual totals", "2018-2023", "National, division and district in selected reports", "Collected for validation", "Annual reconciliation and weekly spot checks", WHO_2023_SITREP, "Published charts are not a machine-readable historical weekly table"],
        ["OpenDengue v1.3", "DOI-backed curated database", "Monthly/annual for Bangladesh", "1980-2025, varying resolution", "National", "Collected", "Independent comparison only", OPENDENGUE_DOI, "2019, 2021, 2022 and 2023 totals differ from DGHS benchmarks"],
        ["Project Tycho Bangladesh dengue", "DOI-backed curated database", "Historical counts", "1980-2006", "National", "Identified; login required", "Long-run context only", "https://www.tycho.pitt.edu/dataset/BD.38362002/", "Too old for the current weekly resource-allocation model"],
        ["NASA POWER daily", "US government remote sensing/reanalysis", "Daily weather", "2018-current collected", "Eight divisional-capital point proxies", "Collected", "Pilot weather covariates", "https://power.larc.nasa.gov/docs/services/api/temporal/daily/", "Point proxies; current days can lag"],
        ["CHIRPS", "Research-grade gridded climate database", "Daily/pentadal/monthly rainfall", "1981-near present", "Gridded", "Identified", "Planned area-weighted rainfall", "https://www.chc.ucsb.edu/data/chirps", "Requires Bangladesh division polygons and raster aggregation"],
        ["ERA5-Land", "Copernicus reanalysis", "Hourly climate", "1950-near present", "Gridded", "Identified", "Planned area-weighted temperature/humidity", "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview", "CDS account/API and large downloads required"],
        ["BBS Population and Housing Census 2022", "Government primary", "Census", "2022", "Division/district/upazila", "Identified", "Population denominators and demand scaling", "https://bbs.portal.gov.bd/site/page/b432a7e5-8b4d-4dac-a76c-a9be4e85828c", "Official host timed out during collection; do not substitute unofficial values silently"],
        ["DGHS Facility Registry", "Government primary", "Facility registry", "Current records", "Facility coordinates and administrative areas", "Identified", "Facility set for allocation model", "https://hris.mohfw.gov.bd/public/facility-registry/", "Bulk API requires credentials; public bed/ICU/HDU fields can be missing or stale"],
        ["Hospital operations data", "DGHS/hospital partnership", "Daily/weekly operations", "Prospective", "Facility", "Not public", "Essential allocation constraints", "", "Need occupied beds, staffing, diagnostics, fluids, referral and surge capacity"],
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "database",
            "authority",
            "temporal_resolution",
            "coverage",
            "geography",
            "collection_status",
            "recommended_use",
            "source_url",
            "limitation",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("work/pre2024_sources"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/expanded_weekly"))
    parser.add_argument("--current-weekly", type=Path, default=Path("work/weekly_covariates/dengue_weekly_national.csv"))
    parser.add_argument("--weather-weekly", type=Path, default=Path("work/weekly_covariates/weather_weekly_national_proxy.csv"))
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local = {}
    for key, url in FILES.items():
        suffix = ".zip" if url.endswith(".zip") else ".xlsx"
        path = args.cache_dir / f"{key}{suffix}"
        if key == "daily" and (args.cache_dir / "Daily_Data_2019-2024.xlsx").exists():
            path = args.cache_dir / "Daily_Data_2019-2024.xlsx"
        elif key == "monthly_cases" and (args.cache_dir / "Dengue_Monthly_Infection_Cleaned_2001-2024.xlsx").exists():
            path = args.cache_dir / "Dengue_Monthly_Infection_Cleaned_2001-2024.xlsx"
        elif key == "monthly_deaths" and (args.cache_dir / "Monthly_Death_2019-2024.xlsx").exists():
            path = args.cache_dir / "Monthly_Death_2019-2024.xlsx"
        elif key == "regional" and (args.cache_dir / "DENV_Regional_2019_2024.xlsx").exists():
            path = args.cache_dir / "DENV_Regional_2019_2024.xlsx"
        elif key == "opendengue_temporal" and (args.cache_dir / "Temporal_extract_V1_3.zip").exists():
            path = args.cache_dir / "Temporal_extract_V1_3.zip"
        download(url, path)
        local[key] = path

    daily = reshape_daily(local["daily"])
    monthly = pd.read_excel(local["monthly_cases"])
    monthly["source"] = "Bhowmik et al. pinned monthly data"
    monthly["source_url"] = PAPER_DATA_URL
    monthly["citation_doi"] = PAPER_DOI
    monthly["use_role"] = "seasonal context and annual reconciliation; never disaggregate to weeks"
    monthly_deaths = pd.read_excel(local["monthly_deaths"])
    regional = pd.read_excel(local["regional"])

    opendengue_path = args.output_dir / "opendengue_bangladesh_temporal.csv"
    opendengue = filter_opendengue(local["opendengue_temporal"], opendengue_path)
    history_weekly = aggregate_historical_weekly(daily)
    current_weekly = aggregate_current_weekly(args.current_weekly)
    expanded = build_weekly_frame(history_weekly, current_weekly, args.weather_weekly)
    reconciliation = annual_reconciliation(monthly, opendengue)
    registry = source_registry()

    daily.to_csv(args.output_dir / "dengue_daily_pre2024_recovered.csv", index=False)
    history_weekly.to_csv(args.output_dir / "dengue_weekly_pre2024_recovered.csv", index=False)
    expanded.to_csv(args.output_dir / "dengue_weekly_model_expanded.csv", index=False)
    monthly.to_csv(args.output_dir / "dengue_monthly_2001_2024.csv", index=False)
    monthly_deaths.to_csv(args.output_dir / "dengue_monthly_deaths_2019_2024.csv", index=False)
    regional.to_csv(args.output_dir / "dengue_regional_2019_2024.csv", index=False)
    reconciliation.to_csv(args.output_dir / "annual_source_reconciliation.csv", index=False)
    registry.to_csv(args.output_dir / "source_registry.csv", index=False)

    manifest = {
        "weekly_rows": int(len(expanded)),
        "weekly_target_available": int(expanded["target_available"].sum()),
        "weekly_target_complete": int(expanded["target_complete"].sum()),
        "weekly_usable_target_weather": int(expanded["usable_target_weather"].sum()),
        "weekly_usable_lagged_model": int(expanded["usable_for_lagged_model"].sum()),
        "coverage_start": expanded["week_start"].min().date().isoformat(),
        "coverage_end": expanded["week_end"].max().date().isoformat(),
        "week_definition": "DGHS Sunday-Saturday; week 1 contains 1 January",
        "pinned_commit": PINNED_COMMIT,
        "paper_doi": PAPER_DOI,
        "opendengue_doi": OPENDENGUE_DOI,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
