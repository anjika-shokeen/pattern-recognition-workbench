.PHONY: install test quick benchmark app notebook report clean
install:
	pip install -r requirements.txt && pip install -e .
test:
	python -m pytest
quick:                      # ~2 minute smoke run of the whole protocol
	python scripts/run_experiments.py --quick --out results_quick
benchmark:                  # full protocol (about 25 minutes on 2 CPU cores)
	python scripts/run_experiments.py --config configs/default.yaml --out results
app:
	streamlit run app/streamlit_app.py
notebook:
	python notebooks/make_notebook.py
report:
	python report/build_report.py
clean:
	rm -rf results_quick .pytest_cache build dist *.egg-info
