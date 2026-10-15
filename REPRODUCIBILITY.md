# Reproducibility guide

## Supported reproduction levels

### 1. Verify the frozen release

Run `python3 src/verify_repository.py`. This checks required repository files, validates the Sunday–Saturday calendar example, confirms the principal forecast/allocation values, and checks every entry in `MANIFEST.sha256` when that manifest is present.

### 2. Regenerate manuscript assets offline

Run `make assets`. It reads committed files in `outputs/` and `data/processed/` and regenerates all PDF/PNG figures, LaTeX row fragments, numerical macros, and the sample data file. It does not access the internet.

### 3. Rebuild the analyses offline

Run `make full`. The dependency order is forecast benchmark, spatial-policy backtest, joint forecast-allocation backtest, latent-demand sensitivity, revision audit, and manuscript assets.

The release contains the processed and intermediate inputs expected by these programs. The full rebuild may take substantial time and can create large division- and week-level simulation files that are excluded from the release archive.

### 4. Refresh public sources

Collection scripts are separate from the reproduction targets because source revisions would no longer reproduce the frozen article. They include `src/scrape_dghs_dengue.py`, `src/collect_weekly_covariates.py`, `src/collect_historical_dengue.py`, and `src/collect_who_sitreps.py`. These require internet access and do not bypass access controls.

## Environment

The analysis was last checked with Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1, Matplotlib 3.9.4, and SciPy 1.13.1. Minimum compatible package versions are listed in `requirements.txt`. The browser dependency is needed only for the optional DGHS acquisition script.

## Determinism

Bootstrap and model seeds are fixed in source. Minor floating-point differences can occur across operating systems and library releases, but any material difference should be investigated before publication.

## LaTeX

The manuscript uses generated row fragments in `manuscript/tables/`. Run `make assets` before `make paper`. A complete TeX Live/MacTeX installation with `latexmk`, `natbib`, `siunitx`, `threeparttable`, and `microtype` is required.
