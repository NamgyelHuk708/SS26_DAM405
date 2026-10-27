# Practical 6: Managing the ML Lifecycle with MLflow

DAM405 Machine Learning Operations.

This project trains a Random Forest classifier on scikit-learn's Wine
dataset and uses MLflow to take it through the full lifecycle:
**track → compare → register → load**.

## Project structure

| File | Purpose |
|------|---------|
| `train.py` | Tasks 2–3: trains 9 models (3 tree counts × 3 depths) and logs each as an MLflow run |
| `register_best.py` | Task 5: finds the best run and registers its model as `wine-classifier` |
| `predict.py` | Task 6: loads the registered model by name and predicts on test data |
| `requirements.txt` | Pinned dependency versions |

## Requirements

- Python 3.10 or newer
- pip

## How to run

### 1. Set up the environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
mlflow --version
```

### 2. Train and track the models

```bash
python train.py
```

This creates 9 runs in the `wine-classifier` experiment. Run metadata is
stored in `mlflow.db` and model files in `mlruns/`.

### 3. Compare runs in the MLflow UI

```bash
mlflow ui
```

Open http://localhost:5000, select the `wine-classifier` experiment and
sort by the `accuracy` column. (If port 5000 is busy, use
`mlflow ui --port 5050`.) Press `Ctrl+C` to stop the UI.

### 4. Register the best model

```bash
python register_best.py
```

The best run is chosen by highest accuracy; ties are broken by preferring
fewer trees and a shallower depth. Running it again creates a new version.

### 5. Load the registered model and predict

```bash
python predict.py
```

### 6. (Optional) Serve the model as a REST API

```bash
mlflow models serve -m "models:/wine-classifier/latest" -p 5001 --env-manager local
```

In a second terminal:

```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"inputs": [[13.2,1.78,2.14,11.2,100,2.65,2.76,0.26,1.28,4.38,1.05,3.4,1050]]}'
```

## Notes

- With `random_state=42`, every configuration reaches 100% accuracy on
  the test split, because the Wine dataset is small and well separated.
  That is why `register_best.py` uses model simplicity as a tie-breaker.
- MLflow 3.x saves scikit-learn models with the `skops` format, which
  rejects tree models unless the `sklearn.tree._tree.Tree` type is marked
  as trusted. `train.py` does this because we trained the model ourselves.