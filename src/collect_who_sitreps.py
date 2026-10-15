#!/usr/bin/env python3
"""Download official WHO Bangladesh 2023 dengue situation reports.

The script follows only report-page links from the WHO dengue index and only
downloads PDF links hosted by cdn.who.int. It does not bypass access controls.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/phase3_latent_demand"
RAW = WORK / "raw"
INDEX_URL = "https://www.who.int/bangladesh/emergencies/dengue-update-2023"
USER_AGENT = "Mozilla/5.0 (compatible; academic reproducibility collection)"


def fetch(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    index = fetch(INDEX_URL).decode("utf-8", errors="replace")
    report_pages = sorted(set(re.findall(
        r'href="([^"]*dengue-situation-report-(\d+)-[^"]+)"',
        index,
        flags=re.I,
    )), key=lambda pair: int(pair[1]))

    manifest = {
        "index_url": INDEX_URL,
        "reports": [],
        "limitations": (
            "Official WHO PDFs only. Extracted text is retained for audit; table parsing "
            "requires field-level validation because PDF reading order can differ from layout."
        ),
    }
    for page_href, issue_text in report_pages:
        issue = int(issue_text)
        page_url = urljoin(INDEX_URL, html.unescape(page_href))
        page_bytes = fetch(page_url)
        page_html = page_bytes.decode("utf-8", errors="replace")
        pdf_candidates = re.findall(
            r'href\s*=\s*["\']?([^"\'\s>]+\.pdf(?:\?[^"\'\s>]*)?)',
            page_html,
            flags=re.I,
        )
        pdf_candidates = [html.unescape(urljoin(page_url, value)) for value in pdf_candidates]
        pdf_candidates = [value for value in pdf_candidates if "cdn.who.int" in value]
        if not pdf_candidates:
            manifest["reports"].append({
                "issue": issue,
                "page_url": page_url,
                "status": "no official PDF link found",
            })
            continue
        pdf_url = pdf_candidates[0]
        pdf_bytes = fetch(pdf_url)
        pdf_path = RAW / f"who_sitrep_{issue}.pdf"
        pdf_path.write_bytes(pdf_bytes)

        reader = PdfReader(pdf_path)
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        text_path = RAW / f"who_sitrep_{issue}.txt"
        text_path.write_text(text, encoding="utf-8")
        manifest["reports"].append({
            "issue": issue,
            "page_url": page_url,
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path.relative_to(ROOT)),
            "pdf_sha256": digest(pdf_bytes),
            "page_count": len(reader.pages),
            "text_path": str(text_path.relative_to(ROOT)),
            "text_sha256": digest(text.encode("utf-8")),
            "contains_hospital_table_marker": bool(
                re.search(r"Hospitals? in (Dhaka|Rest of the Country)", text, flags=re.I)
            ),
            "status": "downloaded",
        })

    (WORK / "who_sitrep_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "report_pages_found": len(report_pages),
        "downloaded": sum(row.get("status") == "downloaded" for row in manifest["reports"]),
        "hospital_table_markers": [
            row["issue"] for row in manifest["reports"]
            if row.get("contains_hospital_table_marker")
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
