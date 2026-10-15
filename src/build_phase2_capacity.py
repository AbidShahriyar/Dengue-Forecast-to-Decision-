#!/usr/bin/env python3
"""Build a restartable, capacity-aware dengue resource-planning dataset.

The forecast target is DGHS reported hospital admissions.  Resource quantities
therefore describe hospital workload, plus transparent sensitivity uplifts for
admission-eligible demand that may not be captured in the reported admissions.
The uplift grid is not an estimate of under-reporting.

The script deliberately keeps the 2025 DGHS facility table as context.  Its
approved beds and annual bed-occupancy rates are not treated as a current
inventory of staffed or empty dengue beds.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase2_capacity"
RAW = WORK / "raw"
OUTPUTS = ROOT / "outputs"
FACILITY_YEAR = 2025
ACCESSED_DATE = "2026-10-13"

FACILITY_TYPES = {
    29: "Upazila Health Complex/Office",
    28: "District/General Hospital",
    5: "Medical College Hospital",
    2: "Postgraduate Institute/Specialized Hospital",
}

CLINICAL_SCENARIOS = {
    "low": {"length_of_stay_days": 4.0, "iv_liters_per_admission": 2.0},
    "central": {"length_of_stay_days": 5.0, "iv_liters_per_admission": 4.5},
    "high": {"length_of_stay_days": 6.0, "iv_liters_per_admission": 6.0},
}

CAPTURE_UPLIFTS = [0.0, 0.10, 0.25, 0.50]
IV_BAG_LITERS = 0.5
SELECTED_FORECAST_LEVEL = "upper"
SELECTED_CAPTURE_UPLIFT = 0.25
SELECTED_CLINICAL_SCENARIO = "central"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the four public DGHS 2025 facility pages before parsing.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(facility_type: int, year: int = FACILITY_YEAR) -> str:
    return (
        "https://dashboard.dghs.gov.bd/pages/"
        f"dashboard_hospital_services_details.php?type={facility_type}&year={year}"
    )


def raw_path(facility_type: int, year: int = FACILITY_YEAR) -> Path:
    return RAW / f"type_{facility_type}_{year}.html"


def download_facility_pages(refresh: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for facility_type in FACILITY_TYPES:
        destination = raw_path(facility_type)
        if destination.exists() and destination.stat().st_size > 10_000 and not refresh:
            continue
        request = Request(
            source_url(facility_type),
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
        )
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        if len(payload) < 10_000:
            raise RuntimeError(
                f"DGHS returned only {len(payload)} bytes for type {facility_type}; "
                "the source may be unavailable or access-limited."
            )
        destination.write_bytes(payload)


class FacilityTableParser(HTMLParser):
    """Extract the first HTML table whose id is 'example1'."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_table = False
        self.depth = 0
        self.in_cell = False
        self.cell_fragments: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        if tag == "table" and attributes.get("id") == "example1" and not self.in_table:
            self.in_table = True
            self.depth = 1
            return
        if not self.in_table:
            return
        if tag == "table":
            self.depth += 1
        elif tag == "tr":
            self.current_row = []
        elif tag in {"td", "th"}:
            self.in_cell = True
            self.cell_fragments = []

    def handle_data(self, data: str) -> None:
        if self.in_table and self.in_cell:
            self.cell_fragments.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag in {"td", "th"} and self.in_cell:
            value = re.sub(r"\s+", " ", html.unescape("".join(self.cell_fragments))).strip()
            self.current_row.append(value)
            self.in_cell = False
            self.cell_fragments = []
        elif tag == "tr" and self.current_row:
            self.rows.append(self.current_row)
            self.current_row = []
        elif tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self.in_table = False


def parse_facility_page(path: Path, facility_type: int) -> pd.DataFrame:
    parser = FacilityTableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    expected = [
        "HRM Code",
        "Organization Name",
        "Division",
        "District",
        "Approved bed no.",
        "Bed occupancy rate",
    ]
    if not parser.rows or parser.rows[0] != expected:
        raise RuntimeError(f"Expected DGHS facility table not found in {path}")
    records = [row for row in parser.rows[1:] if len(row) == len(expected)]
    frame = pd.DataFrame(records, columns=expected).rename(
        columns={
            "HRM Code": "hrm_code",
            "Organization Name": "organization_name",
            "Division": "division",
            "District": "district",
            "Approved bed no.": "approved_beds",
            "Bed occupancy rate": "bed_occupancy_rate_pct",
        }
    )
    frame["hrm_code"] = frame["hrm_code"].astype(str)
    frame["approved_beds"] = pd.to_numeric(frame["approved_beds"], errors="coerce")
    frame["bed_occupancy_rate_pct"] = pd.to_numeric(
        frame["bed_occupancy_rate_pct"], errors="coerce"
    )
    frame["division"] = frame["division"].replace({"Barishal": "Barisal"})
    frame["facility_type_code"] = facility_type
    frame["facility_type"] = FACILITY_TYPES[facility_type]
    frame["source_year"] = FACILITY_YEAR
    frame["source_url"] = source_url(facility_type)
    frame["source_accessed_date"] = ACCESSED_DATE
    return frame


def facility_context(facilities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = facilities[
        facilities["approved_beds"].gt(0)
        & facilities["bed_occupancy_rate_pct"].notna()
    ].copy()
    valid["occupied_bed_equivalent"] = (
        valid["approved_beds"] * valid["bed_occupancy_rate_pct"] / 100.0
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        usable = group[
            group["approved_beds"].gt(0)
            & group["bed_occupancy_rate_pct"].notna()
        ]
        denominator = usable["approved_beds"].sum()
        weighted_bor = (
            (usable["approved_beds"] * usable["bed_occupancy_rate_pct"]).sum()
            / denominator
            if denominator > 0
            else np.nan
        )
        return pd.Series(
            {
                "facility_count": len(group),
                "facilities_with_approved_beds": int(group["approved_beds"].gt(0).sum()),
                "facilities_with_bor": int(group["bed_occupancy_rate_pct"].notna().sum()),
                "approved_public_beds": group["approved_beds"].sum(min_count=1),
                "bed_weighted_annual_bor_pct": weighted_bor,
                "facilities_bor_at_or_above_100_pct": int(
                    group["bed_occupancy_rate_pct"].ge(100).sum()
                ),
            }
        )

    division = (
        facilities.groupby("division", sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    category = (
        facilities.groupby(["division", "facility_type"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    for frame in (division, category):
        frame["source_year"] = FACILITY_YEAR
        frame["interpretation"] = (
            "Historical public-facility context only; approved beds are not staffed/usable "
            "beds and annual BOR is not current free-bed inventory."
        )
    return division, category


def load_demand_scenarios() -> pd.DataFrame:
    path = OUTPUTS / "dengue_division_allocation_scenarios.csv"
    frame = pd.read_csv(path)
    required = {
        "scenario_id",
        "national_forecast_level",
        "share_source_week",
        "division",
        "division_share",
        "national_admissions",
        "division_demand_units",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"Missing allocation scenario fields: {sorted(missing)}")
    return frame


def build_resource_scenarios(base: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for uplift in CAPTURE_UPLIFTS:
        for clinical_name, clinical in CLINICAL_SCENARIOS.items():
            frame = base.copy()
            frame["capture_uplift_fraction"] = uplift
            frame["capture_scenario_note"] = (
                "Illustrative sensitivity uplift for admission-eligible demand not captured "
                "by reported admissions; not an estimated under-reporting rate."
            )
            frame["clinical_scenario"] = clinical_name
            frame["length_of_stay_days"] = clinical["length_of_stay_days"]
            frame["iv_liters_per_admission"] = clinical["iv_liters_per_admission"]
            frame["planning_admissions"] = frame["division_demand_units"] * (1.0 + uplift)
            frame["bed_days"] = frame["planning_admissions"] * frame["length_of_stay_days"]
            frame["average_occupied_beds"] = frame["bed_days"] / 7.0
            frame["iv_liters"] = (
                frame["planning_admissions"] * frame["iv_liters_per_admission"]
            )
            frame["iv_500ml_equivalent_bags"] = frame["iv_liters"] / IV_BAG_LITERS
            frame["resource_scenario_id"] = (
                frame["scenario_id"]
                + f"_u{int(round(100 * uplift)):02d}_{clinical_name}"
            )
            parts.append(frame)
    result = pd.concat(parts, ignore_index=True)
    columns = ["resource_scenario_id"] + [
        column for column in result.columns if column != "resource_scenario_id"
    ]
    return result[columns]


def build_selected_planning_tables(
    base: pd.DataFrame, capacity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = base[base["national_forecast_level"].eq(SELECTED_FORECAST_LEVEL)].copy()
    week_order = (
        selected[["share_source_week"]]
        .drop_duplicates()
        .assign(
            week_number=lambda x: pd.to_numeric(
                x["share_source_week"].str.extract(r"(\d+)")[0], errors="coerce"
            )
        )
        .sort_values("week_number")
    )
    latest_week = str(week_order.iloc[-1]["share_source_week"])
    latest = selected[selected["share_source_week"].eq(latest_week)].copy()
    robust = (
        selected.groupby("division", as_index=False)
        .agg(
            robust_max_division_share=("division_share", "max"),
            robust_share_source_week=(
                "division_share",
                lambda series: selected.loc[series.idxmax(), "share_source_week"],
            ),
        )
    )
    latest = latest[
        ["division", "division_share", "national_admissions", "share_source_week"]
    ].rename(columns={"division_share": "latest_complete_week_share"})
    plan = latest.merge(robust, on="division", how="left").merge(
        capacity, on="division", how="left"
    )
    clinical = CLINICAL_SCENARIOS[SELECTED_CLINICAL_SCENARIO]
    national = float(latest["national_admissions"].iloc[0])
    planning_national = national * (1.0 + SELECTED_CAPTURE_UPLIFT)
    plan["selected_forecast_level"] = SELECTED_FORECAST_LEVEL
    plan["capture_uplift_fraction"] = SELECTED_CAPTURE_UPLIFT
    plan["clinical_scenario"] = SELECTED_CLINICAL_SCENARIO
    plan["length_of_stay_days"] = clinical["length_of_stay_days"]
    plan["iv_liters_per_admission"] = clinical["iv_liters_per_admission"]
    plan["latest_share_planning_admissions"] = (
        planning_national * plan["latest_complete_week_share"]
    )
    plan["robust_envelope_planning_admissions"] = (
        planning_national * plan["robust_max_division_share"]
    )
    for prefix in ["latest_share", "robust_envelope"]:
        plan[f"{prefix}_bed_days"] = (
            plan[f"{prefix}_planning_admissions"] * clinical["length_of_stay_days"]
        )
        plan[f"{prefix}_average_occupied_beds"] = plan[f"{prefix}_bed_days"] / 7.0
        plan[f"{prefix}_iv_liters"] = (
            plan[f"{prefix}_planning_admissions"]
            * clinical["iv_liters_per_admission"]
        )
        plan[f"{prefix}_iv_500ml_equivalent_bags"] = (
            plan[f"{prefix}_iv_liters"] / IV_BAG_LITERS
        )
    plan["capacity_comparison_warning"] = (
        "Do not subtract the 2025 public approved-bed count from forecast bed need; "
        "the source is historical, excludes private capacity, and is not staffed availability."
    )
    plan = plan.sort_values("robust_envelope_planning_admissions", ascending=False)

    national_summary = pd.DataFrame(
        [
            {
                "selected_forecast_level": SELECTED_FORECAST_LEVEL,
                "reported_admissions_forecast": national,
                "capture_uplift_fraction": SELECTED_CAPTURE_UPLIFT,
                "planning_admissions": planning_national,
                "clinical_scenario": SELECTED_CLINICAL_SCENARIO,
                "length_of_stay_days": clinical["length_of_stay_days"],
                "bed_days": planning_national * clinical["length_of_stay_days"],
                "average_occupied_beds": planning_national
                * clinical["length_of_stay_days"]
                / 7.0,
                "iv_liters_per_admission": clinical["iv_liters_per_admission"],
                "iv_liters": planning_national * clinical["iv_liters_per_admission"],
                "iv_500ml_equivalent_bags": planning_national
                * clinical["iv_liters_per_admission"]
                / IV_BAG_LITERS,
                "latest_share_week": latest_week,
                "robust_envelope_share_sum": robust["robust_max_division_share"].sum(),
                "robust_envelope_premium_fraction": robust[
                    "robust_max_division_share"
                ].sum()
                - 1.0,
                "status": "Illustrative stress-planning case, not a minimum clinical stock order.",
            }
        ]
    )

    extremes = []
    for label, level, uplift, clinical_name in [
        ("observed-workload low", "lower", 0.0, "low"),
        ("selected planning", SELECTED_FORECAST_LEVEL, SELECTED_CAPTURE_UPLIFT, SELECTED_CLINICAL_SCENARIO),
        ("stress high", "upper", 0.50, "high"),
    ]:
        row = base[base["national_forecast_level"].eq(level)].iloc[0]
        total = float(row["national_admissions"]) * (1.0 + uplift)
        parameters = CLINICAL_SCENARIOS[clinical_name]
        extremes.append(
            {
                "case": label,
                "forecast_level": level,
                "reported_admissions_forecast": float(row["national_admissions"]),
                "capture_uplift_fraction": uplift,
                "planning_admissions": total,
                "clinical_scenario": clinical_name,
                "length_of_stay_days": parameters["length_of_stay_days"],
                "bed_days": total * parameters["length_of_stay_days"],
                "average_occupied_beds": total
                * parameters["length_of_stay_days"]
                / 7.0,
                "iv_liters_per_admission": parameters["iv_liters_per_admission"],
                "iv_liters": total * parameters["iv_liters_per_admission"],
                "iv_500ml_equivalent_bags": total
                * parameters["iv_liters_per_admission"]
                / IV_BAG_LITERS,
            }
        )
    return plan, national_summary, pd.DataFrame(extremes)


def build_assumptions() -> pd.DataFrame:
    rows = [
        (
            "forecast_target",
            "DGHS reported hospital admissions",
            "The forecast does not represent all dengue infections.",
            "outputs/dengue_biology_current_forecast.csv",
        ),
        (
            "selected_forecast_level",
            SELECTED_FORECAST_LEVEL,
            "The 80% upper forecast is used for the illustrative planning case.",
            "outputs/dengue_biology_current_forecast.csv",
        ),
        (
            "capture_uplift_fraction",
            SELECTED_CAPTURE_UPLIFT,
            "Illustrative sensitivity value; not an estimate of denied admissions or under-reporting.",
            "author-defined sensitivity grid",
        ),
        (
            "length_of_stay_days",
            CLINICAL_SCENARIOS[SELECTED_CLINICAL_SCENARIO]["length_of_stay_days"],
            "Median 5 days; low/high use the reported IQR of 4-6 days.",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11627391/",
        ),
        (
            "iv_liters_per_admission",
            CLINICAL_SCENARIOS[SELECTED_CLINICAL_SCENARIO]["iv_liters_per_admission"],
            "Median-based planning value; low/high use the reported IQR of 2-6 L. "
            "The cohort was selected Group B/C patients in three Dhaka tertiary hospitals.",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11627391/",
        ),
        (
            "iv_bag_liters",
            IV_BAG_LITERS,
            "500-mL-equivalent is a unit conversion, not a clinical prescription or pack-size recommendation.",
            "unit definition",
        ),
        (
            "staffing",
            "not estimated",
            "Requires local workload activities, time standards, rostered productive hours, and skill mix.",
            "https://www.who.int/tools/wisn",
        ),
        (
            "capacity_context",
            f"DGHS public hospital services {FACILITY_YEAR}",
            "Approved beds and annual BOR only; excludes private capacity and does not measure current staffed/empty beds.",
            "https://dashboard.dghs.gov.bd/pages/dashboard_hospital_services.php",
        ),
    ]
    return pd.DataFrame(rows, columns=["parameter", "selected_value", "interpretation", "source"])


def build_source_registry() -> pd.DataFrame:
    rows = [
        (
            "DGHS dengue dashboard",
            "weekly reported hospital admissions; national and division series",
            "used",
            "2024-2026 extracted series",
            "Admissions are realized reported hospital workload, not community incidence or admission-eligible latent demand.",
            "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php",
        ),
        (
            "DGHS Hospital Services",
            "facility, division, district, approved beds, annual bed-occupancy rate",
            "used as context",
            "latest retrievable complete detail period: 2025; 2026 detail pages returned no records",
            "Public-facility context; no live staffed/usable beds or private capacity. A plain request returned 'Goodbye'; an ordinary browser user agent succeeded. No protection was bypassed.",
            "https://dashboard.dghs.gov.bd/pages/dashboard_hospital_services.php",
        ),
        (
            "2019 Bangladesh Group B/C hospitalized cohort",
            "length of stay and IV-fluid utilization",
            "used for sensitivity scenarios",
            "median LOS 5 days (IQR 4-6); median IV fluid 4.5 L (IQR 2-6)",
            "Selected three-tertiary-hospital Dhaka cohort; medians are not national means.",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC11627391/",
        ),
        (
            "WHO WISN",
            "workload-based staffing method",
            "method retained; staff counts withheld",
            "activity time standards applied to workload components",
            "Bangladesh dengue activity times and productive hours are not available in this package.",
            "https://www.who.int/tools/wisn",
        ),
        (
            "Bangladesh National Guideline for Clinical Management of Dengue",
            "clinical management protocol",
            "reference only",
            "official national guideline page/PDF",
            "Patient-specific clinical fluid orders cannot be inferred from an aggregate weekly forecast.",
            "https://dghs.gov.bd/pages/static-pages/6922e05c933eb65569e26afe",
        ),
        (
            "DGHS Facility Registry/HRIS",
            "facility registry and filled/vacant posts",
            "next-stage source",
            "public facility profiles",
            "A reproducible bulk extract and staff-presence/roster validation are still needed.",
            "https://hris.mohfw.gov.bd/public/facility-registry/",
        ),
        (
            "Bangladesh Health Facility Survey 2022",
            "facility readiness and service availability",
            "next-stage source",
            "national facility survey",
            "Cross-sectional and not a weekly stock/capacity feed.",
            "https://buhn.org/uploads/files/resources/reports/national/11_Bangladesh%20Health%20Facility%20Survey%202022.pdf",
        ),
        (
            "Required operational inputs",
            "usable dengue beds, IV inventory, deliveries/lead times, staff rosters/activity times, transfers, diversions/denials",
            "missing/blocking constrained optimization",
            "facility-week data needed",
            "Without these, the model estimates requirements and pre-positioning envelopes but cannot optimize a real stock budget.",
            "not yet available",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["source", "variables", "use_status", "coverage", "limitation", "url"],
    )


def build_report(
    facilities: pd.DataFrame,
    division_context: pd.DataFrame,
    national: pd.DataFrame,
    extremes: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    n = national.iloc[0]
    latest_total = selected["latest_share_planning_admissions"].sum()
    robust_total = selected["robust_envelope_planning_admissions"].sum()
    high_bor = int(facilities["bed_occupancy_rate_pct"].ge(100).sum())
    valid_bor = int(facilities["bed_occupancy_rate_pct"].notna().sum())
    lines = [
        "# Capacity-aware dengue resource-planning prototype",
        "",
        "## What is now workable",
        "",
        "The one-week admission forecast can now be translated into bed-days, average occupied-bed equivalents, IV-fluid litres, and 500-mL-equivalent bags. The operational target remains DGHS reported hospital admissions. A separate 0%, 10%, 25%, and 50% uplift grid represents possible admission-eligible demand not captured in those reports; those uplifts are sensitivity tests, not estimates.",
        "",
        "## Selected illustrative planning case",
        "",
        f"- Forecast: 80% upper one-week admissions forecast = {n['reported_admissions_forecast']:,.1f}.",
        f"- Incomplete-capture sensitivity: +{100*n['capture_uplift_fraction']:.0f}%.",
        f"- Planning admissions = {n['planning_admissions']:,.1f}.",
        f"- Length of stay = {n['length_of_stay_days']:.0f} days, producing {n['bed_days']:,.0f} bed-days or {n['average_occupied_beds']:,.0f} average occupied-bed equivalents.",
        f"- IV planning coefficient = {n['iv_liters_per_admission']:.1f} L per admission, producing {n['iv_liters']:,.0f} L or {math.ceil(n['iv_500ml_equivalent_bags']):,} 500-mL-equivalent bags.",
        f"- Latest-share allocation sums to {latest_total:,.1f} planning admissions. The robust three-week divisional envelope sums to {robust_total:,.1f}, a {100*n['robust_envelope_premium_fraction']:.1f}% spatial reserve premium because each division receives its maximum recent share.",
        "",
        "This is not a clinical prescribing tool or a final purchase order. The clinical coefficients are median-based scenarios from a selected 2019 Group B/C cohort in three Dhaka tertiary hospitals, not national means.",
        "",
        "## Sensitivity range",
        "",
    ]
    for row in extremes.itertuples(index=False):
        lines.append(
            f"- {row.case}: {row.planning_admissions:,.1f} planning admissions; "
            f"{row.bed_days:,.0f} bed-days ({row.average_occupied_beds:,.0f} average occupied beds); "
            f"{row.iv_liters:,.0f} L IV fluid ({math.ceil(row.iv_500ml_equivalent_bags):,} 500-mL equivalents)."
        )
    lines.extend(
        [
            "",
            "## Capacity evidence and the admission-censoring issue",
            "",
            f"The public DGHS 2025 pages yielded {len(facilities):,} facilities and {division_context['approved_public_beds'].sum():,.0f} approved public beds across four facility classes. {high_bor} of {valid_bor} facilities reporting BOR were at or above 100%. This supports treating capacity pressure as plausible, but it does not identify how many dengue patients were denied, diverted, treated elsewhere, or omitted from surveillance.",
            "",
            "Accordingly, under-capture is modeled by partial identification: conclusions are checked over a declared uplift set rather than pretending there is a known correction factor. The 2025 approved-bed/BOR table is not subtracted from dengue bed needs because it is annual, all-cause, public-only, and does not report live staffed or free beds.",
            "",
            "## What remains blocked",
            "",
            "A real constrained allocation optimization still needs facility-week stocks and capacities: usable dengue beds, current IV inventory and replenishment lead times, staff rosters plus activity-time standards, transfer limits, and counts of diversions/denied admissions. Until those arrive, this phase provides defensible requirement scenarios and a robust pre-positioning envelope, not a claim about the exact number of new beds, doctors, or nurses to deploy.",
            "",
            "## Proposed research workflow and novelty test",
            "",
            "1. Freeze the one-week rolling-origin forecast and its calibrated interval before tuning allocation decisions.",
            "2. Translate every historical forecast origin into bed-day and IV-fluid scenarios using the declared clinical and capture-sensitivity grids.",
            "3. Compare spatial policies retrospectively: latest-week shares, pooled recent shares, and the three-week robust maximum-share envelope.",
            "4. Score decision performance, not only forecast error: realized shortage, unused reserve, worst-division service fraction, and allocation stability.",
            "5. Report conclusions that survive all plausible capture uplifts as partially identified; report conclusions that change with uplift as sensitivity-dependent.",
            "6. Add facility-week stocks, staffed capacity, replenishment, and transfers when obtained, then solve the actual constrained allocation problem and compare it with simple proportional policies.",
            "",
            "The publishable contribution is the evaluated chain from calibrated weekly admission uncertainty to equitable, capacity-aware resource decisions under incomplete admission capture. Forecast-to-resource multiplication by itself is not novel enough.",
            "",
            "## Reproducibility",
            "",
            "Run `python3 work/phase2_capacity/build_phase2_capacity.py` to rebuild from cached official pages. Add `--refresh` to re-download the four public DGHS 2025 detail pages. No anti-bot protection is bypassed; a normal browser user agent is used, and any failed/blocked response stops the refresh rather than being circumvented.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = arguments()
    WORK.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    download_facility_pages(args.refresh)
    facilities = pd.concat(
        [parse_facility_page(raw_path(code), code) for code in FACILITY_TYPES],
        ignore_index=True,
    )
    if facilities["hrm_code"].duplicated().any():
        duplicates = facilities.loc[
            facilities["hrm_code"].duplicated(keep=False), "hrm_code"
        ].tolist()
        raise RuntimeError(f"Duplicate HRM codes across facility classes: {duplicates[:10]}")
    division_context, category_context = facility_context(facilities)
    base = load_demand_scenarios()
    scenarios = build_resource_scenarios(base)
    selected, national, extremes = build_selected_planning_tables(base, division_context)
    assumptions = build_assumptions()
    sources = build_source_registry()

    output_frames = {
        "dengue_capacity_facilities_2025.csv": facilities,
        "dengue_capacity_division_context_2025.csv": division_context,
        "dengue_capacity_division_category_context_2025.csv": category_context,
        "dengue_phase2_resource_scenarios.csv": scenarios,
        "dengue_phase2_selected_planning_scenario.csv": selected,
        "dengue_phase2_national_planning_case.csv": national,
        "dengue_phase2_sensitivity_range.csv": extremes,
        "dengue_phase2_assumptions.csv": assumptions,
        "dengue_phase2_data_availability.csv": sources,
    }
    for name, frame in output_frames.items():
        frame.to_csv(OUTPUTS / name, index=False)

    report = build_report(facilities, division_context, national, extremes, selected)
    (OUTPUTS / "dengue_phase2_capacity_report.md").write_text(report, encoding="utf-8")

    input_paths = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [
            *(raw_path(code) for code in FACILITY_TYPES),
            OUTPUTS / "dengue_division_allocation_scenarios.csv",
            OUTPUTS / "dengue_biology_current_forecast.csv",
        ]
    }
    output_names = [f"outputs/{name}" for name in output_frames] + [
        "outputs/dengue_phase2_capacity_report.md"
    ]
    manifest = {
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(),
        "forecast_target": "DGHS reported hospital admissions",
        "facility_context_year": FACILITY_YEAR,
        "inputs": input_paths,
        "parameters": {
            "capture_uplifts": CAPTURE_UPLIFTS,
            "clinical_scenarios": CLINICAL_SCENARIOS,
            "iv_bag_liters": IV_BAG_LITERS,
            "selected_forecast_level": SELECTED_FORECAST_LEVEL,
            "selected_capture_uplift": SELECTED_CAPTURE_UPLIFT,
            "selected_clinical_scenario": SELECTED_CLINICAL_SCENARIO,
        },
        "outputs": output_names,
        "limitations": [
            "Capture uplifts are sensitivity values, not estimated under-reporting rates.",
            "Facility capacity is historical public approved beds/BOR, not live staffed availability.",
            "Clinical coefficients are median-based scenarios from a selected tertiary-hospital cohort.",
            "Staff numbers are withheld pending workload activity and time standards.",
        ],
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint = f"""# Phase 2 capacity-planning checkpoint

Last updated: {datetime.now().astimezone().isoformat()}

Status: complete for requirement scenarios; real stock-constrained allocation remains blocked by facility-week inventory/capacity data.

## Rebuild

- Cached public source pages: `python3 work/phase2_capacity/build_phase2_capacity.py`
- Refresh official DGHS capacity pages: `python3 work/phase2_capacity/build_phase2_capacity.py --refresh`
- Rebuild workbook after CSV refresh: run `node work/phase2_capacity/build_phase2_workbook.mjs`

## Model boundary

- Target: reported dengue hospital admissions, not infections.
- Under-capture: 0%, 10%, 25%, 50% sensitivity grid; not estimated.
- Operational resources estimated: bed-days, average occupied-bed equivalents, IV litres, and 500-mL-equivalent bags.
- Staffing: deliberately not numbered without WISN activity/time standards and rostered productive hours.
- 2025 DGHS capacity: historical public-facility context only; never interpreted as live available beds.
"""
    (WORK / "CHECKPOINT.md").write_text(checkpoint, encoding="utf-8")

    print(
        json.dumps(
            {
                "facilities": len(facilities),
                "divisions": len(division_context),
                "resource_scenario_rows": len(scenarios),
                "selected_planning_rows": len(selected),
                "latest_capacity_year": FACILITY_YEAR,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
