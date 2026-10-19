# Runbook: running DAM405 PA3 step by step

This guide takes you through the whole project in order. Each step lists:

- **What & why**: what the step does, and the file that does it.
- **Run**: the exact commands.
- **Expected output**: what you should see. The numbers come from a verification run on
  2026-10-19. Your run IDs and timestamps will differ; the FLAML numbers may vary slightly.
- **📸 Screenshots**: what to capture and which report section it supports.
  ⭐ = essential for the report. The others are supporting evidence; take them if you can.

After every screenshot, add a row to `reports/evidence_log.md`:

```
| S03_split_summary.png | Stratified split sizes + positive rates | make data | 2026-10-19 |
```

Save screenshots in `reports/images/` and name them as listed here (e.g. `S03_split_summary.png`).

---

## Before you start

**Terminals.** Steps 7–8 need several terminals open at once. All of them must start like this:

```bash
cd ~/Desktop/SEM_7/DAM405/Assignmnet/dam405_Assignmnet_3/dam405-pa3
source .venv/bin/activate
```

You should see `(.venv)` at the start of the prompt.

| Terminal | Used for |
|---|---|
| T1 | Main commands (steps 0–6, 9–10) |
| T2 | MLflow UI (`make mlflow-ui`); leave it running from step 2 on |
| T3 | The API server (`make serve`); leave it running in steps 7–8 |

**Ports.** The API runs on **8010**, because port 8000 is used by another program on this
machine. The MLflow UI runs on **5000**.

**Screenshot tips.** Make the terminal font large enough to read, and clear the terminal
(`clear`) before running the command you want to capture.

---

## Step 0: Environment check

**What & why:** proves the environment is reproducible. Python is 3.11, all package versions
are exact pins, and the logger writes structured JSON lines.

**Run (T1):**

```bash
python --version
python -c "import sklearn, flaml, xgboost, mlflow, shap, fastapi; print('sklearn', sklearn.__version__, '| flaml', flaml.__version__, '| xgboost', xgboost.__version__, '| mlflow', mlflow.__version__, '| shap', shap.__version__, '| fastapi', fastapi.__version__)"
pytest tests/test_logging.py -v
```

**Expected output:**

```
Python 3.11.14
sklearn 1.5.2 | flaml 2.3.1 | xgboost 1.7.6 | mlflow 2.17.2 | shap 0.46.0 | fastapi 0.115.4
tests/test_logging.py::test_writes_json_lines PASSED
tests/test_logging.py::test_exception_is_logged PASSED
tests/test_logging.py::test_no_duplicate_handlers PASSED
============ 3 passed in 0.03s ============
```

📸 ⭐ **S01_environment.png**: the `(.venv)` prompt, the versions line and `3 passed`.
*Report: Setup / Environment.*

---

## Step 1: Load and split the data

**What & why:** `src/data/load_data.py` downloads the Adult dataset (or uses the cached copy),
strips whitespace, turns `?` into missing values and maps the target to 0/1.
`src/data/preprocess.py` makes a **stratified** 70/10/20 train/val/test split, which keeps the
same share of `>50K` in each split. It also saves the raw training rows as
`data/reference/reference.parquet`, the baseline for drift detection.

**Run (T1):**

```bash
clear
make data
```

**Expected output:**

```
INFO | data.load | Loading cached dataset from .../data/raw/adult.csv
INFO | data.load | Dataset shape: 48842 rows x 15 cols
INFO | data.load | Class balance: <=50K (0) = 0.761, >50K (1) = 0.239
INFO | data.load | Missing values per column: {'workclass': 2799, 'occupation': 2809, 'native-country': 857}
...
INFO | data.preprocess | Saved train split:  34188 rows, positive rate 0.2393 -> .../train.parquet
INFO | data.preprocess | Saved val   split:   4885 rows, positive rate 0.2393 -> .../val.parquet
INFO | data.preprocess | Saved test  split:   9769 rows, positive rate 0.2393 -> .../test.parquet
INFO | data.preprocess | Saved drift reference (raw training features) -> .../reference.parquet
INFO | data.preprocess | Preprocessor check: 14 raw features -> 105 model features after one-hot encoding
```

The data loader runs twice (`make data` calls it, and so does `preprocess`); that's expected.

📸 ⭐ **S02_raw_data.png**: the shape `48842 x 15`, the class balance `0.761 / 0.239` and the
missing-value counts.
*Report: Data. The 76/24 imbalance is why ROC-AUC and F1 matter more than accuracy.*

📸 ⭐ **S03_split_summary.png**: the three `Saved ... split` lines (identical 0.2393 rates prove
stratification worked) and the `14 raw -> 105 model features` line.
*Report: Data preparation.*

**Commit:** `feat: data loading, stratified split and drift reference`

---

## Step 2: Baseline model + MLflow tracking

**What & why:** `src/training/train_baseline.py` trains a Logistic Regression inside a sklearn
`Pipeline` (shared preprocessor + model). It logs to MLflow: parameters, val/test metrics,
confusion-matrix and ROC plots, and the model with its input signature. This is the score the
AutoML model has to beat.

**Run (T1):**

```bash
clear
make baseline
```

**Expected output** (takes about 10–40 s):

```
INFO | training.baseline | MLflow experiment 'dam405-pa3-adult-income' (id 1)
INFO | training.baseline | Fitted baseline in 0.70s
INFO | training.baseline | val metrics: {'val_accuracy': 0.8528, ..., 'val_roc_auc': 0.9057, ...}
INFO | training.baseline | test metrics: {'test_accuracy': 0.8522, 'test_precision': 0.7414, 'test_recall': 0.5873, 'test_f1': 0.6554, 'test_roc_auc': 0.9039, 'test_log_loss': 0.3217}
INFO | training.baseline | Baseline run <run_id> finished: test_roc_auc=0.9039
```

A few `alembic` INFO lines appear on the very first run; they come from MLflow creating its
database.

**Start the MLflow UI (T2)** and leave it running:

```bash
make mlflow-ui
```

Open http://127.0.0.1:5000 → click the experiment **dam405-pa3-adult-income** on the left →
click the run **baseline_logreg**.

📸 ⭐ **S04_mlflow_baseline_params.png**: the run page, **Overview**, showing the **Parameters**
table (C, max_iter, random_seed, n_train…) and the **Tags** (model_type, stage).
*Report: Experiment tracking.*

📸 ⭐ **S05_mlflow_baseline_metrics.png**: the same page, scrolled to the **Metrics** table
(`test_roc_auc 0.9039`, `test_f1`, `test_accuracy`, `fit_time_seconds`…).
*Report: Experiment tracking / Results.*

📸 **S06_mlflow_baseline_artifacts.png**: the **Artifacts** tab with the `model/` folder
(MLmodel, model.pkl, requirements.txt) and `plots/` expanded; click one ROC curve to preview it.
*Report: Experiment tracking.*

**Commit:** `feat: logistic regression baseline tracked in MLflow`

---

## Step 3: AutoML with FLAML

**What & why:** `src/training/train_automl.py` fits the preprocessor on the training data, then
FLAML tries lgbm, xgboost, random forest, extra trees and L1 logistic regression for **180
seconds**, optimising validation ROC-AUC. The winning model is wrapped with the fitted
preprocessor into one sklearn Pipeline, evaluated on the test set exactly like the baseline,
and logged to MLflow. The run also logs the best config, the time it took to find the best
model, and the full search history CSV.

**Run (T1):**

```bash
clear
make automl
```

**Expected output** (takes about 3.5–4 minutes). FLAML prints its search progress:

```
INFO | training.automl | Transformed features: 105 columns
INFO | training.automl | Starting FLAML: budget=180s, metric=roc_auc, estimators=['lgbm', 'xgboost', 'rf', 'extra_tree', 'lrl1']
[flaml.automl.logger: ...] {2258} INFO - iteration 73, current learner xgboost
[flaml.automl.logger: ...] {2442} INFO -  at 94.4s,	estimator xgboost's best error=0.0677,	best estimator xgboost's best error=0.0677
...
INFO | training.automl | FLAML best: xgboost (val roc_auc=0.9323) found after 33.2s; config={'n_estimators': ..., 'max_leaves': 4, 'learning_rate': 0.0599, ...}
INFO | training.automl | val metrics: {..., 'val_roc_auc': 0.9323, ...}
INFO | training.automl | test metrics: {'test_accuracy': 0.8751, 'test_precision': 0.7948, 'test_recall': 0.6446, 'test_f1': 0.7119, 'test_roc_auc': 0.9292, 'test_log_loss': 0.275}
INFO | training.automl | AutoML run <run_id> finished
```

FLAML's "error" is `1 - ROC-AUC`, so error 0.0677 means validation ROC-AUC 0.9323. Your
best estimator may be lgbm instead of xgboost; both are fine.

📸 ⭐ **S07_flaml_terminal.png**: the end of the output: a few `flaml.automl.logger` lines
plus the `FLAML best: ...` line and the test metrics.
*Report: AutoML.*

In the MLflow UI (T2), refresh the experiment page.

📸 ⭐ **S08_mlflow_runs_list.png**: the experiment's runs table showing **both** runs
(`automl_flaml`, `baseline_logreg`). To show the scores, click **Columns** and tick
`test_roc_auc` and `test_f1`.
*Report: Experiment tracking.*

📸 **S09_automl_run_detail.png**: the `automl_flaml` run → Overview: parameters (time_budget,
metric, estimator_list, best_estimator) and metrics (`best_loss`, `time_to_best_seconds`,
`test_roc_auc`…).
*Report: AutoML.*

📸 **S10_automl_artifacts.png**: the Artifacts tab with `best_config.json` open, plus
`flaml_search_history.csv`, `flaml/flaml.log`, `model/` and `plots/` visible.
*Report: AutoML.*

📸 ⭐ **S11_mlflow_compare.png**: on the runs table, tick both runs → **Compare**. Capture
the metrics comparison table (baseline vs AutoML, side by side).
*Report: Results (baseline vs AutoML).*

**Commit:** `feat: FLAML AutoML search tracked in MLflow`

---

## Step 4: Register the champion model

**What & why:** `src/training/register_model.py` ranks all runs by `test_roc_auc`, registers
the best one as **adult-income-classifier**, gives that version the alias **@champion**, and
adds a description and tags. The API loads `models:/adult-income-classifier@champion`, so
promoting a different model later only means moving the alias, with no code change.

**Run (T1):**

```bash
clear
make register
```

**Expected output:**

```
INFO | training.register | Leaderboard by test_roc_auc:
INFO | training.register |   1. automl_flaml     test_roc_auc=0.9292  (run ...)
INFO | training.register |   2. baseline_logreg  test_roc_auc=0.9039  (run ...)
INFO | training.register | Registered adult-income-classifier version 1 from run 'automl_flaml' (test_roc_auc=0.9292) with alias @champion
```

📸 **S12a_register_terminal.png** (optional): the leaderboard + "Registered ..." lines.

In the MLflow UI, click **Models** in the top bar.

📸 ⭐ **S12_models_tab.png**: the Models list with **adult-income-classifier**, its latest
version and the `@champion` alias.
*Report: Model registry.*

📸 ⭐ **S13_champion_version.png**: click the model → **Version 1**. Capture the alias
`@champion`, the description ("automl_flaml (flaml_automl) selected on … test_roc_auc =
0.9292"), the tags, and the source run link.
*Report: Model registry.*

**Commit:** `feat: register best model as champion in MLflow registry`

---

## Step 5: SHAP explainability

**What & why:** `src/explain/shap_explain.py` loads the champion **from the registry**, takes 1000
test rows, and uses SHAP's TreeExplainer to compute how much each feature pushes each
prediction up or down. One-hot names such as `cat__sex_Female` are mapped back to
`sex = Female`. The plots show original feature values (e.g. age 53, not the scaled 0.75).
All figures are also logged to the champion's MLflow run, under `shap/`.

**Run (T1):**

```bash
clear
make explain
ls reports/figures/
```

**Expected output:**

```
INFO | explain.shap | Loaded adult-income-classifier@champion (version 1, run ...)
INFO | explain.shap | Explaining 1000 test rows x 105 features
INFO | explain.shap | Top features: {'marital-status = Married-civ-spouse': 1.02, 'age': 0.80, 'capital-gain': 0.49}
INFO | explain.shap | Saved .../shap_summary_beeswarm.png
...
INFO | explain.shap | positive example: test row 1277, P(>50K)=0.749
INFO | explain.shap | negative example: test row 4807, P(>50K)=0.028
INFO | explain.shap | Logged 7 SHAP figures to MLflow run ...
```

Open the PNGs in `reports/figures/` (double-click them in VS Code). These figures go
straight into the report: either screenshot them or insert the PNG files directly.

📸 ⭐ **S14_shap_beeswarm.png** (`shap_summary_beeswarm.png`): global view. Each dot is one
person. Red = high feature value; right of zero = pushes toward `>50K`.
*Report: Explainability (global). Discuss: being married, older age and capital gains increase
the predicted probability.*

📸 ⭐ **S15_shap_bar.png** (`shap_bar.png`): mean |SHAP| ranking of features.
*Report: Explainability (global importance).*

📸 **S16_shap_dependence.png** (`shap_dependence_2_age.png`, plus the other two): how one
feature's effect changes with its value. Age shows an inverted U: the effect rises up to
about 45–55, then falls.
*Report: Explainability.*

📸 ⭐ **S17_shap_waterfalls.png** (`shap_waterfall_positive.png` + `shap_waterfall_negative.png`):
local explanations of one `>50K` and one `<=50K` prediction. Read them from E[f(X)] (the average)
up to f(x) (this person); each bar is one feature's push.
*Report: Explainability (local). Explain both people in plain English.*

📸 (optional) the MLflow run `automl_flaml` → Artifacts → `shap/` folder, showing the figures
are tracked with the model.

**Commit:** `feat: SHAP global and local explanations for champion`

---

## Step 6: Drift detection tests

**What & why:** `src/serving/drift.py` compares recent live inputs with the training reference.
- **Numeric features:** Kolmogorov–Smirnov test; a feature has drifted if p < 0.05.
- **Categorical features:** Population Stability Index; a feature has drifted if PSI > 0.2.
  Categories under 1% of training data are pooled into `OTHER`, so rare countries don't
  cause false alarms.
- **Overall drift:** at least 2 features must drift.

The unit tests prove the detector stays quiet on same-distribution data and fires on shifted data.

**Run (T1):**

```bash
clear
pytest tests/test_drift.py -v
```

**Expected output:**

```
tests/test_drift.py::test_same_distribution_no_drift PASSED
tests/test_drift.py::test_shifted_distribution_detects_drift PASSED
tests/test_drift.py::test_psi_is_zero_for_identical_series PASSED
tests/test_drift.py::test_psi_handles_unseen_category PASSED
tests/test_drift.py::test_psi_pools_rare_categories PASSED
tests/test_drift.py::test_detector_insufficient_data_then_ok PASSED
============ 6 passed ============
```

📸 ⭐ **S18_drift_tests.png**: all 6 tests PASSED.
*Report: Drift detection / Testing.*

---

## Step 7: Serve the model with FastAPI

**What & why:** `src/serving/app.py` loads the champion at startup. For every request, a
middleware assigns a `request_id`, times the request, records metrics and writes **one JSON
log line** to `logs/service.log`. Invalid input is rejected by the Pydantic schema
(`src/serving/schemas.py`) with a 422 error.

**Run (T3)** and leave it running:

```bash
make serve
```

**Expected output:**

```
INFO:     Waiting for application startup.
INFO | serving | Service starting
INFO | serving | Loaded model models:/adult-income-classifier@champion (version 1)
INFO | serving | Service ready: model version 1, drift window 200
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8010 (Press CTRL+C to quit)
```

Open http://127.0.0.1:8010/docs in the browser.

📸 ⭐ **S19_fastapi_docs.png**: the Swagger page listing `GET /health`, `POST /predict`,
`GET /metrics`, `GET /drift`.
*Report: Serving.*

**Successful prediction:** in /docs, expand **POST /predict** → **Try it out**. The example
body is pre-filled → **Execute**.

📸 ⭐ **S20_predict_success.png**: response code **200** and a body with the fields
`prediction` (0 or 1), `label`, `probability`, `model_version: "1"` and `request_id`.
The response headers also show the same `x-request-id`.
*Report: Serving.*

**Failed request:** in the same box, change `"age": 39` to `"age": "abc"` → **Execute**.

📸 ⭐ **S21_predict_422.png**: response code **422**, with the error saying `age` must be a
valid integer.
*Report: Serving / Error handling.*

(Terminal alternative for S20 and S21, run in T1:)

```bash
curl -s -X POST http://127.0.0.1:8010/predict -H "Content-Type: application/json" -d '{"age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13, "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40, "native-country": "United-States"}' | jq
curl -s -X POST http://127.0.0.1:8010/predict -H "Content-Type: application/json" -d '{"age": "abc"}' | jq
```

📸 **S22_uvicorn_console.png**: T3 after those requests, showing the human-readable log
lines, e.g. `INFO | serving | POST /predict -> 200 (5.1 ms)` and
`ERROR | serving | POST /predict -> 422 (2.0 ms)`.
*Report: Logging.*

---

## Step 8: Simulate traffic, then check monitoring and drift

**What & why:** `scripts/simulate_traffic.py` sends traffic in three phases:
1. 300 normal requests sampled from the test set.
2. 300 **drifted** requests: age +15, hours ×1.5, more capital gains, 70% `Self-emp-inc`,
   70% `Never-married`. This batch is also saved to `data/processed/drifted_batch.csv`.
3. 10 malformed requests.

After each phase it prints the `/drift` result; at the end it prints `/metrics`.

**Run (T1),** with the server still running in T3:

```bash
clear
make simulate
```

**Expected output:**

```
API healthy: {'status': 'ok', 'model_loaded': True, 'model_name': 'adult-income-classifier', 'model_version': '1'}

Phase 1: 300 normal requests
  [normal] sent 300/300  status counts: {200: 300}

--- GET /drift after NORMAL traffic ---
drift_detected=False  n_drifted=1  drifted_features=['age']

Phase 2: 300 drifted requests (saved to .../drifted_batch.csv)
  [drifted] sent 300/300  status counts: {200: 300}

--- GET /drift after DRIFTED traffic ---
drift_detected=True  n_drifted=5  drifted_features=['age', 'capital-gain', 'hours-per-week', 'workclass', 'marital-status']

Phase 3: 10 malformed requests
  [malformed] sent 10/10  status counts: {422: 10}

=== Summary: status codes per phase ===
  normal     {200: 300}
  drifted    {200: 300}
  malformed  {422: 10}

--- GET /metrics ---
{ "request_count": 614, "error_count": 10, "error_rate": 0.0163,
  "latency_ms": {"avg": 6.3, "p50": 4.9, "p95": 9.0, "p99": 49.6, ...}, ... }
```

During normal traffic, 0 or 1 feature may be flagged by chance. That's why overall drift
needs at least 2 features, and why the result is still `drift_detected=False`.

📸 ⭐ **S25_drift_normal.png**: the `GET /drift after NORMAL traffic` block
(`drift_detected=False`).
*Report: Drift detection.*

📸 ⭐ **S26_drift_drifted.png**: the `GET /drift after DRIFTED traffic` block
(`drift_detected=True`, the 5 features). Optionally also open
http://127.0.0.1:8010/drift in the browser to show the full per-feature JSON (KS statistic,
p-value, PSI per feature).
*Report: Drift detection.*

📸 ⭐ **S24_metrics.png**: open http://127.0.0.1:8010/metrics in the browser (Firefox shows
the JSON nicely). Capture `request_count`, `error_count`, `error_rate`, the latency
percentiles and `predictions_by_class`.
*Report: Monitoring.*

**Inspect the structured log file (T1):**

```bash
clear
echo "INFO: $(grep -c '"level": "INFO"' logs/service.log)  WARNING: $(grep -c '"level": "WARNING"' logs/service.log)  ERROR: $(grep -c '"level": "ERROR"' logs/service.log)"
grep -m1 '"POST /predict -> 200' logs/service.log | jq
grep -m1 '"level": "ERROR"' logs/service.log | jq
grep -m1 'Data drift detected' logs/service.log | jq
```

**Expected output:** counts roughly like `INFO: 612  WARNING: 17  ERROR: 10`, then three
pretty-printed JSON records:
- **INFO:** a normal request, with `request_id`, `endpoint`, `status_code`, `latency_ms`.
- **ERROR:** a 422 request.
- **WARNING:** the drift alert, with `drifted_features`.

📸 ⭐ **S23_service_log_levels.png**: the level counts and the three JSON records (INFO,
ERROR, WARNING).
*Report: Logging (structured JSON with request_id, latency, levels).*

```bash
clear
grep 'Data drift detected' logs/service.log | jq -c '{timestamp, level, message}'
```

📸 **S27_drift_warnings.png**: the drift WARNING lines over time.
*Report: Drift detection / Logging.*

When finished, stop the server with **Ctrl+C** in T3.

**Commit:** `feat: FastAPI service with monitoring, logging and drift detection`

---

## Step 9: Offline drift report + full test suite

**What & why:** `scripts/offline_drift_report.py` runs the same KS/PSI checks on a whole batch
file (the drifted batch from Step 8) and saves a per-feature table plus reference-vs-live
histograms. This is how you would audit a day of production data offline.

**Run (T1):**

```bash
clear
make drift-report
```

**Expected output:**

```
       feature        type test  statistic  p_value  threshold  drifted
     workclass categorical  psi     2.7520      NaN       0.20     True
marital-status categorical  psi     1.1320      NaN       0.20     True
           age     numeric   ks     0.4154 0.000000       0.05     True
  capital-gain     numeric   ks     0.4030 0.000000       0.05     True
hours-per-week     numeric   ks     0.7107 0.000000       0.05     True
     education categorical  psi     0.0547      NaN       0.20    False
     ...
Drifted features: 5/14 -> overall drift: True
Saved report to .../reports/drift_report.csv
Saved histograms to .../reports/figures/drift_histograms.png
```

`p_value` is `NaN` for categorical features because PSI is a distance, not a test.

📸 ⭐ **S30_offline_drift_table.png**: the table + the "overall drift: True" line.
*Report: Drift detection.*

📸 ⭐ **S29_drift_histograms.png** (`reports/figures/drift_histograms.png`): reference vs live
distributions for age, hours-per-week and capital-gain. The shift is visible.
*Report: Drift detection.*

**Full test suite and training log:**

```bash
clear
make test
```

**Expected output:** `17 passed` (3 logging + 6 drift + 8 API). The API tests use a fake
model, so they don't need the server running.

📸 ⭐ **S31_pytest_all.png**: all 17 tests PASSED.
*Report: Testing.*

```bash
clear
jq -c '{timestamp, level, logger, message}' logs/training.log | tail -n 15
```

📸 **S28_training_log.png**: JSON log lines from training (split sizes, metrics, registration).
*Report: Logging.*

**Commit:** `test: drift and API tests; offline drift report`

---

## Step 10: Architecture, README and report

📸 ⭐ **S32_architecture.png**: open `docs/architecture.md` in VS Code and press
**Ctrl+Shift+V** (Markdown preview). If the diagram doesn't render, install the extension
"Markdown Preview Mermaid Support", or paste the `mermaid` block into https://mermaid.live.
*Report: Architecture.*

```bash
clear
tree -L 2 -I '.venv|__pycache__|mlartifacts'
```

📸 **S33_repo_tree.png**: the project structure. *Report: Appendix / Project structure.*

**Finish the README** (`README.md`):
- **Results:** fill in the table from S05 / S09 / S11.
- **Evidence:** link the screenshots.
- **Limitations:** add your own observations to the list.
- **AI-use declaration:** base it on `docs/ai_use_log.md`.

**Commit:** `docs: results, evidence and AI-use declaration`

---

## Screenshot checklist

| # | File | Report section | ⭐ |
|---|---|---|---|
| S01 | environment | Setup | ⭐ |
| S02 | raw_data | Data | ⭐ |
| S03 | split_summary | Data preparation | ⭐ |
| S04 | mlflow_baseline_params | Experiment tracking | ⭐ |
| S05 | mlflow_baseline_metrics | Experiment tracking | ⭐ |
| S06 | mlflow_baseline_artifacts | Experiment tracking | |
| S07 | flaml_terminal | AutoML | ⭐ |
| S08 | mlflow_runs_list | Experiment tracking | ⭐ |
| S09 | automl_run_detail | AutoML | |
| S10 | automl_artifacts | AutoML | |
| S11 | mlflow_compare | Results | ⭐ |
| S12 | models_tab | Model registry | ⭐ |
| S13 | champion_version | Model registry | ⭐ |
| S14 | shap_beeswarm | Explainability | ⭐ |
| S15 | shap_bar | Explainability | ⭐ |
| S16 | shap_dependence | Explainability | |
| S17 | shap_waterfalls | Explainability | ⭐ |
| S18 | drift_tests | Drift / Testing | ⭐ |
| S19 | fastapi_docs | Serving | ⭐ |
| S20 | predict_success | Serving | ⭐ |
| S21 | predict_422 | Serving | ⭐ |
| S22 | uvicorn_console | Logging | |
| S23 | service_log_levels | Logging | ⭐ |
| S24 | metrics | Monitoring | ⭐ |
| S25 | drift_normal | Drift | ⭐ |
| S26 | drift_drifted | Drift | ⭐ |
| S27 | drift_warnings | Drift / Logging | |
| S28 | training_log | Logging | |
| S29 | drift_histograms | Drift | ⭐ |
| S30 | offline_drift_table | Drift | ⭐ |
| S31 | pytest_all | Testing | ⭐ |
| S32 | architecture | Architecture | ⭐ |
| S33 | repo_tree | Appendix | |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | The venv isn't active: `source .venv/bin/activate`. |
| `FileNotFoundError ... Run make data first` | Run Step 1 first. |
| `make serve`: `address already in use` | A previous server is still running. Stop it with Ctrl+C in its terminal, or change `serving.port` in `config/config.yaml`. |
| Server fails with `CRITICAL ... Could not load model` | Run `make register` first (Step 4). |
| `make simulate`: "API not reachable" | Start `make serve` in T3 first. |
| MLflow UI is empty | It must be started from the project folder with `make mlflow-ui`; refresh the page after training. |
| You want to start completely fresh | Stop the server and MLflow UI, then: `rm -rf mlflow.db mlartifacts logs/*.log reports/figures/*.png reports/*.csv` and redo from Step 1. |
| You re-ran `make automl` / `make register` | That's fine: new runs and a new model version appear, and `@champion` moves to the best run. Mention it in the report if your screenshots show version 2. |
