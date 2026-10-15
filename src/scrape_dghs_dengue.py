#!/usr/bin/env python3
"""Extract public dengue-dashboard data from DGHS using an ordinary browser.

The DGHS page is server-rendered and places its datasets inside inline
Highcharts configuration.  Direct non-browser HTTP requests may receive HTTP
403, so this script uses Playwright with an installed Chrome browser.  It does
not alter the user agent, use stealth plugins, solve challenges, or bypass any
access control.

Example:
    python3 scrape_dghs_dengue.py --years 2025 --out dghs_extract
    python3 scrape_dghs_dengue.py --years 2026 --from-date 2026-09-01 \
        --to-date 2026-09-30 --out dghs_september_2026
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


URL = "https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php"

CHART_TITLES = {
    "affected_case_last_24_hour": "Dengue cases of last 24 hours",
    "death_case_last_24_hour": "Dengue deaths of last 24 hours",
    "div_city_cor_case_last_24_hour": "Division and city-corporation cases of last 24 hours",
    "div_city_cor_death_last_24_hour": "Division and city-corporation deaths of last 24 hours",
    "dengue_affected_by_gender_24": "Gender distribution of affected cases of last 24 hours",
    "dengue_discharged_total_and_24_hours": "Discharged",
    "affected_case_in_year": "Total dengue cases from 1 January to date",
    "death_case_in_year": "Total dengue deaths from 1 January to date",
    "div_city_cor_case_in_year": "Division and city-corporation cases from 1 January to date",
    "div_city_cor_death_in_year": "Division and city-corporation deaths from 1 January to date",
    "monthly_case_and_death_in_year": "Monthly dengue admissions in selected year",
    "monthly_case_and_death_only_in_year": "Monthly dengue deaths in selected year",
    "division_case": "Dengue admissions by division in selected year",
    "division_death": "Dengue deaths by division in selected year",
    "confirmed_case": "Dengue admissions by date",
    "death_case": "Dengue deaths by date",
    "by_week_case": "Comparative dengue admissions by epidemiological week",
    "by_month_case": "Comparative dengue admissions by month",
    "year_case": "Dengue admissions by year",
    "dengue_affected_by_age_group": "Dengue admissions by age group and sex",
    "percentage_of_dengue_affected_by_age_group": "Percentage of admissions by age group and sex",
    "dengue_death_by_age_group": "Dengue deaths by age group and sex",
    "percentage_of_dengue_death_by_age_group": "Percentage of deaths by age group and sex",
    "dengue_affected_by_gender": "Gender distribution of admissions",
    "dengue_death_by_gender": "Gender distribution of deaths",
    "affected_in_division_by_week": "Dengue admissions by division and epidemiological week",
    "death_case_ration_by_week": "Dengue deaths versus admissions by epidemiological week",
}

EXTRACT_JS = r"""
() => {
  const clean = value => (value || '').replace(/\s+/g, ' ').trim();

  function nearbyHeading(element) {
    let node = element;
    for (let depth = 0; node && depth < 7; depth += 1, node = node.parentElement) {
      const candidates = Array.from(node.querySelectorAll('h1,h2,h3,h4,.box-title,.panel-title'));
      const heading = candidates.find(h => clean(h.textContent));
      if (heading) return clean(heading.textContent);
    }
    return '';
  }

  const charts = (window.Highcharts && Highcharts.charts ? Highcharts.charts : [])
    .filter(Boolean)
    .map(chart => {
      const categories = (chart.xAxis && chart.xAxis[0] && chart.xAxis[0].categories) || [];
      return {
        chart_id: chart.renderTo && chart.renderTo.id || '',
        nearby_heading: nearbyHeading(chart.renderTo),
        categories: Array.from(categories, value => clean(String(value))),
        series: chart.series.map(series => ({
          name: clean(series.name),
          type: series.type || '',
          points: series.points.map((point, index) => ({
            index,
            x: Number.isFinite(point.x) ? point.x : null,
            y: Number.isFinite(point.y) ? point.y : null,
            category: categories[index] !== undefined
              ? clean(String(categories[index]))
              : (point.category !== undefined ? clean(String(point.category)) : ''),
            point_name: point.name !== undefined ? clean(String(point.name)) : '',
            percentage: Number.isFinite(point.percentage) ? point.percentage : null
          }))
        }))
      };
    });

  const tables = Array.from(document.querySelectorAll('table')).map((table, tableIndex) => ({
    table_index: tableIndex,
    nearby_heading: nearbyHeading(table),
    rows: Array.from(table.querySelectorAll('tr')).map((row, rowIndex) => ({
      row_index: rowIndex,
      cells: Array.from(row.querySelectorAll('th,td')).map(cell => clean(cell.textContent)),
      is_header: row.querySelectorAll('th').length > 0 || rowIndex === 0
    }))
  }));

  const allText = clean(document.body.innerText);
  const lastUpdatedMatch = allText.match(/Last Updated:\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})/i);
  const selected = document.querySelector('select[name="filter_year"]');
  return {
    page_title: document.title,
    selected_year: selected ? selected.value : '',
    available_years: selected ? Array.from(selected.options).map(o => o.value) : [],
    last_updated: lastUpdatedMatch ? lastUpdatedMatch[1] : '',
    charts,
    tables
  };
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2025"],
        help="Dashboard years to request (default: 2025).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dghs_extract"),
        help="Output directory (default: ./dghs_extract).",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Show Chrome while extracting (headless by default).",
    )
    parser.add_argument(
        "--save-html",
        action="store_true",
        help="Also save the rendered HTML for each selected year.",
    )
    parser.add_argument(
        "--report-date",
        help="Optional YYYY-MM-DD point-in-time dashboard snapshot (one year only).",
    )
    parser.add_argument(
        "--from-date",
        help="Optional YYYY-MM-DD start of the daily-admissions chart range (one year only).",
    )
    parser.add_argument(
        "--to-date",
        help="Optional YYYY-MM-DD end of the daily-admissions chart range (one year only).",
    )
    return parser.parse_args()


def submit_year(page: Any, year: str, report_date: str | None = None) -> int | None:
    current = page.locator('select[name="filter_year"]').input_value()
    if current == year and not report_date:
        return None
    page.select_option('select[name="filter_year"]', year)
    if report_date:
        page.fill('input[name="report_date_filter"]', report_date)
    with page.expect_navigation(wait_until="domcontentloaded", timeout=90_000) as navigation:
        page.locator('input[name="search_filter"]').click()
    response = navigation.value
    return response.status if response else None


def submit_daily_range(page: Any, from_date: str, to_date: str) -> int | None:
    page.fill('input[name="confirm_from"]', from_date)
    page.fill('input[name="confirm_to"]', to_date)
    with page.expect_navigation(wait_until="domcontentloaded", timeout=90_000) as navigation:
        page.locator('input[name="confirm_search"]').click()
    response = navigation.value
    return response.status if response else None


def wait_for_charts(page: Any) -> None:
    page.wait_for_function(
        "window.Highcharts && Highcharts.charts && Highcharts.charts.filter(Boolean).length > 0",
        timeout=45_000,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_chart_points(path: Path, snapshots: list[dict[str, Any]]) -> int:
    columns = [
        "selected_year",
        "last_updated",
        "chart_id",
        "chart_title",
        "series",
        "series_type",
        "point_index",
        "x",
        "category",
        "point_name",
        "value",
        "percentage",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for snapshot in snapshots:
            for chart in snapshot["charts"]:
                title = CHART_TITLES.get(chart["chart_id"], chart["nearby_heading"])
                for series in chart["series"]:
                    for point in series["points"]:
                        writer.writerow(
                            {
                                "selected_year": snapshot["selected_year"],
                                "last_updated": snapshot["last_updated"],
                                "chart_id": chart["chart_id"],
                                "chart_title": title,
                                "series": series["name"],
                                "series_type": series["type"],
                                "point_index": point["index"],
                                "x": point["x"],
                                "category": point["category"],
                                "point_name": point["point_name"],
                                "value": point["y"],
                                "percentage": point["percentage"],
                            }
                        )
                        count += 1
    return count


def write_table_cells(path: Path, snapshots: list[dict[str, Any]]) -> int:
    columns = [
        "selected_year",
        "last_updated",
        "table_index",
        "table_title",
        "row_index",
        "column_index",
        "is_header",
        "value",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for snapshot in snapshots:
            for table in snapshot["tables"]:
                for row in table["rows"]:
                    for column_index, value in enumerate(row["cells"]):
                        writer.writerow(
                            {
                                "selected_year": snapshot["selected_year"],
                                "last_updated": snapshot["last_updated"],
                                "table_index": table["table_index"],
                                "table_title": table["nearby_heading"],
                                "row_index": row["row_index"],
                                "column_index": column_index,
                                "is_header": row["is_header"],
                                "value": value,
                            }
                        )
                        count += 1
    return count


def write_research_sample(path: Path, snapshots: list[dict[str, Any]]) -> int:
    """Write a compact, non-duplicated subset in a tidy research format."""
    columns = [
        "selected_year",
        "granularity",
        "period",
        "geography",
        "age_group",
        "sex",
        "measure",
        "value",
        "unit",
        "source_chart_id",
        "note",
    ]
    rows: list[dict[str, Any]] = []

    def append_chart(
        snapshot: dict[str, Any],
        chart_id: str,
        granularity: str,
        measure: str,
        *,
        geography: bool = False,
        age_sex: bool = False,
        note: str = "",
    ) -> None:
        chart = next((c for c in snapshot["charts"] if c["chart_id"] == chart_id), None)
        if not chart:
            return
        for series in chart["series"]:
            for point in series["points"]:
                period = point["category"] or point["point_name"]
                rows.append(
                    {
                        "selected_year": snapshot["selected_year"],
                        "granularity": granularity,
                        "period": period if not (geography or age_sex) else snapshot["selected_year"],
                        "geography": period if geography else "Bangladesh",
                        "age_group": period if age_sex else "all",
                        "sex": series["name"] if age_sex else "all",
                        "measure": measure,
                        "value": point["y"],
                        "unit": "count",
                        "source_chart_id": chart_id,
                        "note": note,
                    }
                )

    for snapshot in snapshots:
        append_chart(snapshot, "confirmed_case", "day", "admissions")
        append_chart(
            snapshot,
            "death_case",
            "day",
            "deaths",
            note="Dashboard lists dates with reported deaths; zero-death dates are absent.",
        )
        append_chart(snapshot, "monthly_case_and_death_in_year", "month", "admissions")
        append_chart(snapshot, "monthly_case_and_death_only_in_year", "month", "deaths")
        append_chart(
            snapshot,
            "div_city_cor_case_in_year",
            "selected-year total",
            "admissions",
            geography=True,
        )
        append_chart(
            snapshot,
            "div_city_cor_death_in_year",
            "selected-year total",
            "deaths",
            geography=True,
        )
        append_chart(
            snapshot,
            "dengue_affected_by_age_group",
            "selected-year total",
            "admissions",
            age_sex=True,
            note="Raw age labels preserved; validate malformed/overlapping source labels before analysis.",
        )
        append_chart(
            snapshot,
            "dengue_death_by_age_group",
            "selected-year total",
            "deaths",
            age_sex=True,
            note="Raw age labels preserved; validate malformed/overlapping source labels before analysis.",
        )

        weekly = next(
            (c for c in snapshot["charts"] if c["chart_id"] == "death_case_ration_by_week"),
            None,
        )
        if weekly:
            for series in weekly["series"]:
                if series["name"] not in {"Affected (Admitted)", "Death"}:
                    continue
                measure = "admissions" if series["name"] == "Affected (Admitted)" else "deaths"
                for point in series["points"]:
                    rows.append(
                        {
                            "selected_year": snapshot["selected_year"],
                            "granularity": "epidemiological week",
                            "period": point["category"],
                            "geography": "Bangladesh",
                            "age_group": "all",
                            "sex": "all",
                            "measure": measure,
                            "value": point["y"],
                            "unit": "count",
                            "source_chart_id": "death_case_ration_by_week",
                            "note": "Blank week labels are preserved as published.",
                        }
                    )
        else:
            division_week = next(
                (c for c in snapshot["charts"] if c["chart_id"] == "affected_in_division_by_week"),
                None,
            )
            if division_week and division_week["series"]:
                point_count = max(len(s["points"]) for s in division_week["series"])
                for index in range(point_count):
                    points = [s["points"][index] for s in division_week["series"] if index < len(s["points"])]
                    rows.append(
                        {
                            "selected_year": snapshot["selected_year"],
                            "granularity": "epidemiological week",
                            "period": points[0]["category"] if points else "",
                            "geography": "Bangladesh",
                            "age_group": "all",
                            "sex": "all",
                            "measure": "admissions",
                            "value": sum(point["y"] or 0 for point in points),
                            "unit": "count",
                            "source_chart_id": "affected_in_division_by_week",
                            "note": "National value reconstructed as the sum of the eight published division series; the national weekly chart was absent for this selected year.",
                        }
                    )

    if snapshots:
        annual = next(
            (c for c in snapshots[0]["charts"] if c["chart_id"] == "year_case"), None
        )
        if annual:
            for point in annual["series"][0]["points"]:
                rows.append(
                    {
                        "selected_year": point["category"],
                        "granularity": "year",
                        "period": point["category"],
                        "geography": "Bangladesh",
                        "age_group": "all",
                        "sex": "all",
                        "measure": "admissions",
                        "value": point["y"],
                        "unit": "count",
                        "source_chart_id": "year_case",
                        "note": "Annual series exposed on every dashboard view; included once.",
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    requested_years = [str(year) for year in args.years]
    if bool(args.from_date) != bool(args.to_date):
        print("Extraction failed: --from-date and --to-date must be supplied together", file=sys.stderr)
        return 1
    if (args.report_date or args.from_date) and len(requested_years) != 1:
        print("Extraction failed: date filters can be used with exactly one --years value", file=sys.stderr)
        return 1
    for label, value in (
        ("--report-date", args.report_date),
        ("--from-date", args.from_date),
        ("--to-date", args.to_date),
    ):
        if value:
            try:
                datetime.strptime(value, "%Y-%m-%d")
            except ValueError:
                print(f"Extraction failed: {label} must use YYYY-MM-DD", file=sys.stderr)
                return 1
    snapshots: list[dict[str, Any]] = []
    request_log: list[dict[str, Any]] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed, channel="chrome")
            page = browser.new_page()

            def record_request(request: Any) -> None:
                if request.resource_type in {"document", "xhr", "fetch"}:
                    request_log.append(
                        {
                            "resource_type": request.resource_type,
                            "method": request.method,
                            "url": request.url,
                            "post_data": request.post_data,
                        }
                    )

            page.on("request", record_request)
            response = page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
            if response and response.status >= 400:
                raise RuntimeError(f"Dashboard returned HTTP {response.status}")
            wait_for_charts(page)

            available = page.locator('select[name="filter_year"] option').evaluate_all(
                "options => options.map(option => option.value)"
            )
            invalid = [year for year in requested_years if year not in available]
            if invalid:
                raise ValueError(
                    f"Unavailable year(s): {', '.join(invalid)}. "
                    f"Dashboard currently offers: {', '.join(available)}"
                )

            for year in requested_years:
                status = submit_year(page, year, args.report_date)
                if status is not None and status >= 400:
                    raise RuntimeError(f"Year {year} request returned HTTP {status}")
                wait_for_charts(page)
                if args.from_date and args.to_date:
                    range_status = submit_daily_range(page, args.from_date, args.to_date)
                    if range_status is not None and range_status >= 400:
                        raise RuntimeError(
                            f"Daily range request returned HTTP {range_status}"
                        )
                    wait_for_charts(page)
                snapshot = page.evaluate(EXTRACT_JS)
                snapshot["source_url"] = URL
                snapshot["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
                snapshot["navigation_status"] = status if status is not None else response.status
                snapshot["requested_filters"] = {
                    "year": year,
                    "report_date": args.report_date,
                    "from_date": args.from_date,
                    "to_date": args.to_date,
                }
                snapshots.append(snapshot)
                write_json(args.out / f"dashboard_{year}.json", snapshot)
                if args.save_html:
                    (args.out / f"dashboard_{year}.html").write_text(
                        page.content(), encoding="utf-8"
                    )

            browser.close()
    except (PlaywrightTimeoutError, PlaywrightError, RuntimeError, ValueError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1

    chart_rows = write_chart_points(args.out / "chart_points.csv", snapshots)
    table_cells = write_table_cells(args.out / "table_cells.csv", snapshots)
    research_rows = write_research_sample(args.out / "research_sample.csv", snapshots)
    manifest = {
        "source_url": URL,
        "requested_years": requested_years,
        "requested_filters": {
            "report_date": args.report_date,
            "from_date": args.from_date,
            "to_date": args.to_date,
        },
        "available_years": snapshots[0]["available_years"] if snapshots else [],
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "chart_point_rows": chart_rows,
        "table_cell_rows": table_cells,
        "research_sample_rows": research_rows,
        "network_requests": request_log,
        "notes": [
            "Values are hospital-reported/admitted cases shown by the DGHS dashboard.",
            "No XHR or fetch data API was observed; data are embedded in the HTML/Highcharts configuration.",
            "Raw category labels are preserved, including apparent source-data anomalies.",
        ],
    }
    write_json(args.out / "manifest.json", manifest)
    print(
        f"Wrote {len(snapshots)} snapshot(s), {chart_rows} chart points, "
        f"{table_cells} table cells, and {research_rows} research-sample rows to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
