PYTHON ?= python3

.PHONY: verify assets forecast spatial joint censoring audit full paper clean

verify:
	$(PYTHON) src/verify_repository.py

assets:
	MPLBACKEND=Agg MPLCONFIGDIR=/tmp/dengue-mpl XDG_CACHE_HOME=/tmp/dengue-cache $(PYTHON) src/make_manuscript_assets.py

forecast:
	$(PYTHON) src/build_biology_benchmark.py

spatial:
	$(PYTHON) src/backtest_spatial_policies.py

joint:
	$(PYTHON) src/backtest_joint_forecast_allocation.py

censoring:
	$(PYTHON) src/build_phase3_latent_demand.py

audit:
	$(PYTHON) src/build_manuscript_revision_audit.py

full: forecast spatial joint censoring audit assets

paper: assets
	latexmk -pdf -cd manuscript/main.tex

clean:
	latexmk -C -cd manuscript/main.tex
