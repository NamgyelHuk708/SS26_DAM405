# Architecture

End-to-end workflow, from raw data to a monitored, explainable prediction service.
All settings come from `config/config.yaml`.

```mermaid
flowchart TD
    subgraph DATA["1. Data"]
        A["OpenML: Adult dataset v2"] -->|"load_data.py"| B["data/raw/adult.csv"]
        B -->|"preprocess.py"| C["Train / Val / Test splits (70/10/20, stratified)"]
        B -->|"preprocess.py"| R["Reference snapshot (raw training features)"]
    end

    subgraph TRAIN["2. Training and tracking"]
        C --> P["Shared preprocessor: impute + scale / one-hot"]
        P --> L["Baseline: Logistic Regression"]
        P --> F["FLAML AutoML: lgbm, xgboost, rf, extra_tree, lrl1"]
        L --> E["evaluate.py: metrics + ROC / confusion plots"]
        F --> E
        E --> M["MLflow Tracking: params, metrics, artifacts"]
    end

    subgraph REG["3. Registry and explainability"]
        M -->|"register_model.py: best test ROC-AUC"| G["Model Registry: adult-income-classifier, alias champion"]
        G -->|"shap_explain.py"| S["SHAP: beeswarm, bar, dependence, waterfall"]
        S --> FIG["reports/figures"]
        S -->|"log artifacts"| M
    end

    subgraph SERVE["4. Serving and monitoring"]
        G -->|"load champion at startup"| API["FastAPI service (app.py)"]
        U["Client / simulate_traffic.py"] -->|"POST /predict"| API
        API --> MW["Middleware: request_id, latency"]
        MW --> MS["MetricsStore: count, errors, latency p50/p95/p99"]
        API --> D["DriftDetector: KS numeric, PSI categorical"]
        R --> D
        MW --> LOG["logs/service.log (JSON lines)"]
        D -->|"WARNING on drift"| LOG
        MS --> EP1["GET /metrics"]
        D --> EP2["GET /drift"]
        R --> OFF["offline_drift_report.py"]
        OFF --> DR["reports/drift_report.csv"]
    end
```

## Components

| Stage | Module | Output |
|---|---|---|
| Ingest | `src/data/load_data.py` | `data/raw/adult.csv` |
| Split + reference | `src/data/preprocess.py` | `data/processed/*.parquet`, `data/reference/reference.parquet` |
| Baseline | `src/training/train_baseline.py` | MLflow run `baseline_logreg` |
| AutoML | `src/training/train_automl.py` | MLflow run `automl_flaml` (+ search history CSV) |
| Registry | `src/training/register_model.py` | `adult-income-classifier@champion` |
| Explain | `src/explain/shap_explain.py` | `reports/figures/*.png`, artifacts on champion run |
| Serve | `src/serving/app.py` | `/health`, `/predict`, `/metrics`, `/drift` |
| Monitor | `src/serving/monitoring.py`, `drift.py` | `logs/service.log`, endpoint JSON |
