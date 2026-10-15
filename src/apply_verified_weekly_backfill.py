#!/usr/bin/env python3
"""Overlay only externally verified weekly dengue totals on the model frame.

Daily gaps are deliberately left untouched. This script promotes a target week
to complete only when an official or official-secondary source publishes the
same Sunday-Saturday weekly total used by the model.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "work/expanded_weekly/dengue_weekly_model_expanded.csv"
DEFAULT_BACKFILL = PROJECT / "work/backfill/verified_sun_sat_backfill.csv"
DEFAULT_OUTPUT = PROJECT / "work/expanded_weekly/dengue_weekly_model_verified.csv"
DEFAULT_AUDIT = PROJECT / "outputs/dengue_targeted_backfill_audit.csv"
DEFAULT_REPORT = PROJECT / "outputs/dengue_targeted_backfill_report.md"
DEFAULT_MANIFEST = PROJECT / "work/backfill/manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--backfill", type=Path, default=DEFAULT_BACKFILL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.output.parent, args.audit.parent, args.report.parent, args.manifest.parent]:
        path.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, parse_dates=["week_start", "week_end"])
    overlay = pd.read_csv(args.backfill, parse_dates=["week_start", "week_end"])
    frame = frame.sort_values("week_start").reset_index(drop=True)
    frame["target_week_total_verified"] = frame["target_complete"].fillna(False).astype(bool)
    frame["target_observation_basis"] = ""
    frame.loc[frame["target_complete"].fillna(False), "target_observation_basis"] = (
        "sum of seven published daily observations"
    )
    frame["target_backfill_method"] = ""
    frame["target_backfill_limitations"] = ""

    audit_rows: list[dict] = []
    for item in overlay.itertuples(index=False):
        match = frame.index[frame["week_start"].eq(item.week_start)]
        if len(match) != 1:
            raise ValueError(f"Expected one model row for {item.week_start.date()}, found {len(match)}")
        i = int(match[0])
        if frame.at[i, "week_end"] != item.week_end:
            raise ValueError(f"Week-end mismatch for {item.week_start.date()}")
        existing = frame.at[i, "admissions"]
        if pd.notna(existing) and float(existing) > float(item.admissions):
            raise ValueError(
                f"Partial daily sum {existing} exceeds verified weekly total {item.admissions} "
                f"for {item.week_start.date()}"
            )
        audit_rows.append(
            {
                "week_start": item.week_start.date().isoformat(),
                "week_end": item.week_end.date().isoformat(),
                "original_admissions_partial_sum": existing,
                "original_daily_observations": frame.at[i, "target_days_observed"],
                "verified_weekly_admissions": int(item.admissions),
                "difference_from_partial_sum": (
                    float(item.admissions) - float(existing) if pd.notna(existing) else None
                ),
                "source": item.source,
                "source_url": item.source_url,
                "verification_method": item.verification_method,
                "derivation": item.derivation,
                "limitations": item.limitations,
            }
        )
        frame.at[i, "admissions"] = int(item.admissions)
        frame.at[i, "target_available"] = True
        frame.at[i, "target_complete"] = True
        frame.at[i, "target_week_total_verified"] = True
        frame.at[i, "target_observation_basis"] = "externally verified Sunday-Saturday weekly total"
        frame.at[i, "target_backfill_method"] = item.verification_method
        frame.at[i, "target_backfill_limitations"] = item.limitations
        frame.at[i, "target_source"] = item.source
        frame.at[i, "target_source_url"] = item.source_url
        frame.at[i, "coverage_scope"] = "complete published weekly total; daily series remains incomplete"
        frame.at[i, "target_gap_reason"] = ""

    frame["target_complete"] = frame["target_complete"].fillna(False).astype(bool)
    frame["target_available"] = frame["admissions"].notna()
    frame["weather_complete"] = frame["weather_complete"].fillna(False).astype(bool)
    frame["usable_target_weather"] = frame["target_complete"] & frame["weather_complete"]
    complete = frame["target_complete"].astype(int)
    frame["prior_4_target_weeks_complete"] = complete.shift(1).rolling(4, min_periods=4).sum().eq(4)
    frame["usable_for_lagged_model"] = frame["usable_target_weather"] & frame["prior_4_target_weeks_complete"]
    frame.loc[~frame["target_available"], "target_gap_reason"] = "no exact public weekly target recovered"
    frame.loc[
        frame["target_available"] & ~frame["target_complete"], "target_gap_reason"
    ] = "partial daily target with no externally verified weekly total"

    audit = pd.DataFrame(audit_rows)
    frame.to_csv(args.output, index=False)
    audit.to_csv(args.audit, index=False)

    unresolved = frame[~frame["target_complete"]]
    report = f"""# Targeted dengue backfill report

Generated: {datetime.now().astimezone().isoformat()}

## Result

- Verified weekly totals added: **{len(audit)}**.
- Complete weekly targets after overlay: **{int(frame['target_complete'].sum())} of {len(frame)}**.
- Weeks still incomplete: **{len(unresolved)}**.
- Daily observations were not fabricated or redistributed.

The earlier four WHO bulletin totals covered Monday-Sunday intervals and were
therefore not reused after correcting the model to the DGHS Sunday-Saturday
calendar. The 2020-2021 target blackout remains missing because monthly totals
and plotted epidemic curves do not uniquely identify the weekly trajectory.

## Acceptance rule

A week is promoted to complete only when a citable source gives the exact
Sunday-Saturday weekly total, or when it is exactly recoverable from published
cumulative and adjacent weekly totals. Partial daily observations are retained
for audit, but unknown daily values remain unknown.

## Access limitation

The legacy DGHS archive timed out both programmatically and in a normal browser.
No access control or anti-bot protection was bypassed. The newer DGHS report
server requests a mobile number and OTP and was not accessed.
"""
    args.report.write_text(report, encoding="utf-8")

    manifest = {
        "status": "complete",
        "updated_at": datetime.now().astimezone().isoformat(),
        "input": str(args.input),
        "backfill": str(args.backfill),
        "output": str(args.output),
        "audit": str(args.audit),
        "report": str(args.report),
        "verified_weeks": len(audit),
        "complete_weeks": int(frame["target_complete"].sum()),
        "unresolved_weeks": int((~frame["target_complete"]).sum()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
