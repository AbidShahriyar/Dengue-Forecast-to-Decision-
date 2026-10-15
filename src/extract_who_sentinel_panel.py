#!/usr/bin/env python3
"""Extract the WHO 2023 sentinel-hospital weekly dengue tables.

The panel covers WHO situation-report issues 6-12 (epidemiological weeks
39-45). Issue 7 publishes only five Dhaka-city and five rest-of-country rows;
the other six issues publish ten rows in each group. PDF text-extraction rows
are parsed and checked against cumulative week-to-week arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase3_latent_demand"
RAW = WORK / "raw"
OUTPUT = ROOT / "outputs/dengue_who_2023_sentinel_hospital_weekly.csv"
AUDIT = ROOT / "outputs/dengue_who_2023_sentinel_hospital_audit.csv"
MANIFEST = WORK / "who_sitrep_manifest.json"

CANONICAL = {
    "dhaka mch": ("Dhaka Medical College Hospital", "Dhaka city"),
    "dhaka mmch": ("Dhaka Medical College Hospital", "Dhaka city"),
    "mugda mch": ("Mugda Medical College Hospital", "Dhaka city"),
    "sir salimullah mch": ("Sir Salimullah Medical College Hospital", "Dhaka city"),
    "kurmitola general hosp": ("Kurmitola General Hospital", "Dhaka city"),
    "kurmitola general": ("Kurmitola General Hospital", "Dhaka city"),
    "kurmitola gen hosp": ("Kurmitola General Hospital", "Dhaka city"),
    "dncc dedicated hosp": ("DNCC Dedicated Hospital", "Dhaka city"),
    "dncc dedica hosp": ("DNCC Dedicated Hospital", "Dhaka city"),
    "enam mc hosp": ("Enam Medical College and Hospital", "Dhaka city"),
    "shishu hospital inst": ("Bangladesh Shishu Hospital and Institute", "Dhaka city"),
    "shishu hosp inst": ("Bangladesh Shishu Hospital and Institute", "Dhaka city"),
    "square hospitals ltd": ("Square Hospitals Ltd.", "Dhaka city"),
    "asgar ali hospital": ("Asgar Ali Hospital", "Dhaka city"),
    "green life hopital ltd": ("Green Life Hospital Ltd.", "Dhaka city"),
    "green life hospital ltd": ("Green Life Hospital Ltd.", "Dhaka city"),
    "green life hosp ltd": ("Green Life Hospital Ltd.", "Dhaka city"),
    "sbmch barishal": ("Sher-e-Bangla Medical College Hospital", "rest of country"),
    "sher e bangla mch": ("Sher-e-Bangla Medical College Hospital", "rest of country"),
    "faridpur mch": ("Faridpur Medical College Hospital", "rest of country"),
    "chattogram mch": ("Chattogram Medical College Hospital", "rest of country"),
    "khulna mch": ("Khulna Medical College Hospital", "rest of country"),
    "rajshahi mch": ("Rajshahi Medical College Hospital", "rest of country"),
    "mymensingh mch": ("Mymensingh Medical College Hospital", "rest of country"),
    "satkhira mch": ("Satkhira Medical College Hospital", "rest of country"),
    "manikganj mch": ("Manikganj Medical College Hospital", "rest of country"),
    "szrmch bogura": ("Shaheed Ziaur Rahman Medical College Hospital", "rest of country"),
    "shahid zr mch": ("Shaheed Ziaur Rahman Medical College Hospital", "rest of country"),
    "shahid ziaur rahman mch": ("Shaheed Ziaur Rahman Medical College Hospital", "rest of country"),
    "shahid ziaur rah man mch": ("Shaheed Ziaur Rahman Medical College Hospital", "rest of country"),
    "s z rahman mch": ("Shaheed Ziaur Rahman Medical College Hospital", "rest of country"),
    "patuakhali mch": ("Patuakhali Medical College Hospital", "rest of country"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    value = value.replace("‑", "-").replace("–", "-")
    value = value.replace("&", " ")
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).strip().lower()
    return value


def parse_numeric_row(line: str) -> dict | None:
    tokens = line.split()
    cfr_indexes = [index for index, token in enumerate(tokens) if token.endswith("%")]
    if len(cfr_indexes) != 1:
        return None
    cfr_index = cfr_indexes[0]
    if cfr_index < 3 or len(tokens) < cfr_index + 3:
        return None

    cumulative_deaths = tokens[cfr_index - 1]
    if not cumulative_deaths.isdigit():
        return None
    cumulative_case_end = cfr_index - 1
    cumulative_case_start = cumulative_case_end - 1
    if (
        cumulative_case_start - 1 >= 0
        and tokens[cumulative_case_start].isdigit()
        and len(tokens[cumulative_case_start]) == 3
        and tokens[cumulative_case_start - 1].isdigit()
    ):
        cumulative_case_start -= 1
    name = " ".join(tokens[:cumulative_case_start]).strip()
    cumulative_cases_tokens = tokens[cumulative_case_start:cumulative_case_end]
    after = tokens[cfr_index + 1:]
    if len(after) < 2 or not all(token.isdigit() for token in after):
        return None
    weekly_deaths = after[-1]
    weekly_cases_tokens = after[:-1]
    if not name or not cumulative_cases_tokens or not weekly_cases_tokens:
        return None
    return {
        "source_hospital_label": name,
        "cumulative_cases": int("".join(cumulative_cases_tokens)),
        "cumulative_deaths": int(cumulative_deaths),
        "cumulative_cfr": float(tokens[cfr_index].rstrip("%")) / 100.0,
        "weekly_cases": int("".join(weekly_cases_tokens)),
        "weekly_deaths": int(weekly_deaths),
    }


def hospital_segment(lines: list[str]) -> list[str]:
    start = None
    for index, line in enumerate(lines):
        joined = " ".join(lines[index:index + 2]).lower()
        if "hospitals in" in joined and "dhaka city" in joined:
            start = index
            break
    if start is None:
        raise RuntimeError("Hospital table start was not found.")
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith("Contact:")),
        len(lines),
    )
    return lines[start:end]


def extract_issue(issue: int, page_url: str, pdf_url: str) -> list[dict]:
    text_path = RAW / f"who_sitrep_{issue}.txt"
    raw_lines = [re.sub(r"\s+", " ", line).strip() for line in text_path.read_text().splitlines()]
    lines = [line for line in raw_lines if line]
    segment = hospital_segment(lines)
    rows: list[dict] = []
    previous = ""
    for line in segment:
        parsed = parse_numeric_row(line)
        if parsed is not None:
            normalized = normalize_name(parsed["source_hospital_label"])
            if normalized not in CANONICAL and previous:
                joined = f"{previous} {line}"
                candidate = parse_numeric_row(joined)
                if candidate is not None:
                    candidate_normalized = normalize_name(candidate["source_hospital_label"])
                    if candidate_normalized in CANONICAL:
                        parsed = candidate
                        normalized = candidate_normalized
            if normalized in CANONICAL:
                canonical_name, source_group = CANONICAL[normalized]
                rows.append({
                    "issue": issue,
                    "epi_week": issue + 33,
                    "hospital": canonical_name,
                    "source_group": source_group,
                    **parsed,
                    "source_page_url": page_url,
                    "source_pdf_url": pdf_url,
                    "source_agency": "WHO Bangladesh; data source HEOC & Control Room, MIS-DGHS",
                    "quality_note": (
                        "Weekly and cumulative hospitalized dengue cases/deaths transcribed from the "
                        "WHO situation-report table. Issue 7 lists only five hospitals per group."
                    ),
                })
        previous = line
    return rows


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    report_lookup = {row["issue"]: row for row in manifest["reports"] if row.get("status") == "downloaded"}
    all_rows: list[dict] = []
    for issue in range(6, 13):
        report = report_lookup[issue]
        all_rows.extend(extract_issue(issue, report["page_url"], report["pdf_url"]))
    panel = pd.DataFrame(all_rows).sort_values(["epi_week", "source_group", "hospital"]).reset_index(drop=True)
    panel.to_csv(OUTPUT, index=False)

    expected_rows = {6: 20, 7: 10, 8: 20, 9: 20, 10: 20, 11: 20, 12: 20}
    audits = []
    for issue in range(6, 13):
        group = panel[panel["issue"].eq(issue)]
        audits.append({
            "issue": issue,
            "epi_week": issue + 33,
            "parsed_rows": len(group),
            "expected_rows_from_published_table": expected_rows[issue],
            "unique_hospitals": group["hospital"].nunique(),
            "duplicate_hospital_rows": int(group.duplicated("hospital").sum()),
            "row_count_pass": len(group) == expected_rows[issue],
        })
    audit = pd.DataFrame(audits)

    # Independent arithmetic check: this week's cumulative cases should equal
    # last week's cumulative plus current weekly cases when consecutive rows are
    # published. Late corrections may create non-zero differences, so preserve
    # rather than overwrite them.
    ordered = panel.sort_values(["hospital", "epi_week"]).copy()
    ordered["previous_cumulative_cases"] = ordered.groupby("hospital")["cumulative_cases"].shift(1)
    ordered["previous_epi_week"] = ordered.groupby("hospital")["epi_week"].shift(1)
    ordered["cumulative_case_reconciliation_difference"] = (
        ordered["cumulative_cases"]
        - ordered["previous_cumulative_cases"]
        - ordered["weekly_cases"]
    )
    comparable = ordered["epi_week"].eq(ordered["previous_epi_week"] + 1)
    recon = ordered[comparable]
    audit_summary = pd.DataFrame([{
        "issue": "all consecutive pairs",
        "epi_week": "39-45",
        "parsed_rows": len(panel),
        "expected_rows_from_published_table": sum(expected_rows.values()),
        "unique_hospitals": panel["hospital"].nunique(),
        "duplicate_hospital_rows": int(panel.duplicated(["epi_week", "hospital"]).sum()),
        "row_count_pass": len(panel) == sum(expected_rows.values()),
        "consecutive_hospital_week_pairs": len(recon),
        "exact_cumulative_case_reconciliations": int(
            recon["cumulative_case_reconciliation_difference"].eq(0).sum()
        ),
        "maximum_absolute_reconciliation_difference": float(
            recon["cumulative_case_reconciliation_difference"].abs().max()
        ) if len(recon) else None,
    }])
    audit = pd.concat([audit, audit_summary], ignore_index=True)
    audit.to_csv(AUDIT, index=False)

    extraction_manifest = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "audit": str(AUDIT.relative_to(ROOT)),
        "audit_sha256": sha256(AUDIT),
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha256(MANIFEST),
        "issues": list(range(6, 13)),
        "epi_weeks": list(range(39, 46)),
        "row_count": len(panel),
        "important_limit": (
            "Seven peak-season weeks are sufficient for descriptive triangulation only, not for "
            "training or independently validating a hospital-level forecasting model. The tables "
            "contain admissions/deaths, not staffed beds, occupancy, refusals, transfers, or stocks."
        ),
    }
    (WORK / "who_sentinel_extraction_manifest.json").write_text(
        json.dumps(extraction_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(panel.groupby(["issue", "epi_week"]).size().to_string())
    print(f"Rows: {len(panel)}; hospitals: {panel['hospital'].nunique()}")


if __name__ == "__main__":
    main()
