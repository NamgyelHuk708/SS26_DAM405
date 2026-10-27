"""
Task 5: Register the best model in the MLflow Model Registry.

Instead of copying a run ID by hand, this script searches the
experiment for the run with the highest accuracy and registers it.
If several runs tie on accuracy, it prefers the simplest model
(fewest trees, then shallowest depth), which is cheaper to serve.
"""
import mlflow

EXPERIMENT_NAME = "wine-classifier"
MODEL_NAME = "wine-classifier"

# Get every run in the experiment as a pandas DataFrame.
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
if runs.empty:
    raise SystemExit("No runs found. Run train.py first.")

# Parameters are stored as text, so convert them to numbers for sorting.
# A depth of None (unlimited) is treated as the most complex option.
runs["n"] = runs["params.n_estimators"].astype(int)
runs["depth"] = runs["params.max_depth"].replace("None", "999").astype(int)

# Sort: highest accuracy first, then fewest trees, then shallowest depth.
runs = runs.sort_values(
    ["metrics.accuracy", "n", "depth"], ascending=[False, True, True]
)

best = runs.iloc[0]
run_id = best["run_id"]
print(f"Best run: {run_id}")
print(f"  n_estimators={best['params.n_estimators']}, "
      f"max_depth={best['params.max_depth']}, "
      f"accuracy={best['metrics.accuracy']:.4f}")

# Register the model logged under that run. Each call creates a new version.
result = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
print("Registered version:", result.version)