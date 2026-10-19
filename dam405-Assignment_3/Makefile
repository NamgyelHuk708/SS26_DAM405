# DAM405 PA3 — workflow entry points. Run `make help` for a list.
# Host/port and the MLflow store are read from config/config.yaml (no duplication here).

VENV    := .venv
PYTHON  := $(VENV)/bin/python
CFG      = $(PYTHON) -c "from src.config import load_config, mlflow_tracking_uri, mlflow_artifact_location as a; c = load_config(); print($(1))"

HOST    = $(shell $(call CFG,c['serving']['host']))
PORT    = $(shell $(call CFG,c['serving']['port']))
MLFLOW_URI = $(shell $(call CFG,mlflow_tracking_uri(c)))
ARTIFACTS  = $(shell $(call CFG,a(c)))
BATCH  ?= data/processed/drifted_batch.csv

.PHONY: help setup data baseline automl register explain serve simulate drift-report mlflow-ui test clean

help:
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create .venv (Python 3.11) and install pinned requirements
	uv venv --python 3.11 $(VENV) || python3.11 -m venv $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt || $(PYTHON) -m pip install -r requirements.txt

data: ## Download data, split, save processed splits + drift reference
	$(PYTHON) -m src.data.load_data
	$(PYTHON) -m src.data.preprocess

baseline: ## Train + track the Logistic Regression baseline
	$(PYTHON) -m src.training.train_baseline

automl: ## Run FLAML AutoML + track in MLflow
	$(PYTHON) -m src.training.train_automl

register: ## Register the best run as the champion
	$(PYTHON) -m src.training.register_model

explain: ## SHAP global + local explanations for the champion
	$(PYTHON) -m src.explain.shap_explain

serve: ## Start the FastAPI service
	$(PYTHON) -m uvicorn src.serving.app:app --host $(HOST) --port $(PORT)

simulate: ## Send normal, drifted and malformed traffic to the running API
	$(PYTHON) -m scripts.simulate_traffic

drift-report: ## Offline drift report (override batch with BATCH=path.csv)
	$(PYTHON) -m scripts.offline_drift_report --batch $(BATCH)

mlflow-ui: ## Open the MLflow UI on the project's tracking store
	$(VENV)/bin/mlflow ui --backend-store-uri $(MLFLOW_URI) --default-artifact-root $(ARTIFACTS) --port 5000

test: ## Run the test suite
	$(PYTHON) -m pytest -v

clean: ## Remove caches and generated outputs (keeps data/raw, mlflow.db and mlartifacts)
	find . -type d -name __pycache__ -not -path "./$(VENV)/*" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -f logs/*.log logs/*.log.* reports/figures/*.png reports/*.csv
