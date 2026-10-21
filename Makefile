.PHONY: all setup run data rules laya-baseline laya-finetune compare plots serve clean test

PYTHON ?= python3

all: run

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) src/data_generator.py

rules:
	$(PYTHON) src/rules.py

laya-baseline:
	$(PYTHON) src/laya_baseline.py

laya-finetune:
	$(PYTHON) src/laya_finetune.py

compare:
	$(PYTHON) src/comparison.py

plots:
	$(PYTHON) src/plot_results.py

run: data rules laya-baseline laya-finetune compare plots
	@echo "=========================================================="
	@echo " POC Benchmark Pipeline Completed Successfully!"
	@echo " Results available in results/ and rendered at docs/index.html"
	@echo "=========================================================="

demo:
	$(PYTHON) src/demo_scenarios.py

serve:
	$(PYTHON) -m uvicorn salesforce.fastapi_bridge:app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf __pycache__ src/__pycache__ salesforce/__pycache__ .pytest_cache
	rm -rf data/splits/*.csv data/synthetic_cases.csv
	rm -rf results/*.png results/*.csv results/*.md
