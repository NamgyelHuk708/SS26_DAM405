import argparse
import csv
import os
import time

import joblib
from flaml import AutoML
from sklearn.metrics import accuracy_score

from baseline import get_baseline
from data import load_split

# ---- Command-line options (defaults match the practical sheet) ----
parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=60, help="time budget in seconds")
parser.add_argument("--metric", default="accuracy", help="e.g. accuracy, roc_auc")
parser.add_argument("--out", default="automl_model.joblib", help="output model file")
args = parser.parse_args()

# ---- Data and baseline ----
X_train, X_test, y_train, y_test = load_split()
_, base_acc = get_baseline()
print(f"Baseline accuracy: {base_acc:.4f}")

# ---- Task 3: Run AutoML ----
print(f"\nRunning FLAML: metric={args.metric}, time_budget={args.budget}s ...")
automl = AutoML()
start = time.time()
automl.fit(
    X_train,
    y_train,
    task="classification",
    metric=args.metric,
    time_budget=args.budget,
    seed=42,
    log_file_name="flaml.log",
)
elapsed = time.time() - start

# ---- Task 4: Inspect the best model ----
print("\n===== Best model =====")
print("Best estimator :", automl.best_estimator)
print("Best config    :", automl.best_config)
print(f"Best CV score  : {1 - automl.best_loss:.4f}")
print(f"Search time    : {elapsed:.1f}s")

# ---- Task 5: Evaluate on the test set ----
auto_acc = accuracy_score(y_test, automl.predict(X_test))
diff = auto_acc - base_acc
print("\n===== Test-set comparison =====")
print(f"AutoML test accuracy  : {auto_acc:.4f}")
print(f"Baseline test accuracy: {base_acc:.4f}")
print(f"Difference (AutoML - baseline): {diff:+.4f}")

# ---- Task 6: Save the best model ----
joblib.dump(automl, args.out)
print(f"\nSaved {args.out}")

# ---- Keep records for the report ----
with open("results.txt", "w") as f:
    f.write(f"metric={args.metric}\n")
    f.write(f"time_budget={args.budget}\n")
    f.write(f"best_estimator={automl.best_estimator}\n")
    f.write(f"best_config={automl.best_config}\n")
    f.write(f"best_cv_score={1 - automl.best_loss:.4f}\n")
    f.write(f"automl_test_accuracy={auto_acc:.4f}\n")
    f.write(f"baseline_test_accuracy={base_acc:.4f}\n")
    f.write(f"difference={diff:+.4f}\n")
    f.write(f"search_time_seconds={elapsed:.1f}\n")

new_file = not os.path.exists("results.csv")
with open("results.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(["metric", "budget_s", "best_estimator", "cv_score",
                    "automl_test_acc", "baseline_test_acc", "diff", "search_time_s"])
    w.writerow([args.metric, args.budget, automl.best_estimator,
                f"{1 - automl.best_loss:.4f}", f"{auto_acc:.4f}",
                f"{base_acc:.4f}", f"{diff:+.4f}", f"{elapsed:.1f}"])