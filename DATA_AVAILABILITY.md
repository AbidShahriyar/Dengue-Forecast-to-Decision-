# Data availability and provenance

## Public surveillance

The primary recent source is the DGHS Health Emergency Operation Center dengue dashboard:

<https://dashboard.dghs.gov.bd/pages/heoc_dengue_v1.php>

Cached dashboard responses for 2024–2026 are included in `data/raw/dghs/` to make the reported study snapshot auditable. They contain aggregated counts only. Users should verify DGHS terms and current publication requirements before redistributing a release.

The processed national and divisional weekly series used in the article are also included. These frozen files permit offline reproduction; they are not a substitute for the current DGHS dashboard.

## Historical dengue series

Pre-2024 records are reconciled from the data repository cited by Bhowmik et al. (2026), pinned by commit in `src/collect_historical_dengue.py`, and OpenDengue version 1.3. The processed source registry records URLs, DOI identifiers, temporal resolution, and use in the study. Monthly totals are preserved for context and are never downscaled into weekly outcomes.

## Weather

Weather covariates come from the NASA POWER Daily API at eight divisional-capital point locations. These are reproducible proxy points, not polygon-wide divisional averages. Frozen daily and weekly analytical weather files are included for offline analysis; the collection program can retrieve a fresh copy when internet access is available.

## Capacity and clinical coefficients

The public-capacity context comes from 2025 DGHS facility pages and a Transparency International Bangladesh 2023 field report. The clinical scenario coefficients come from a published 2019 cohort in three Dhaka tertiary hospitals. Neither source measures current free dengue beds, inventory, staffing, denied admission, or diversion.

## Reuse warning

The MIT License applies only to original repository code. It does not relicense DGHS, WHO, NASA, OpenDengue, journal, or Transparency International materials. Cite every underlying source and comply with its terms.
