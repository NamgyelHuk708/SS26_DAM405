"""
Tasks 2 & 3: Track training runs with MLflow.

Trains a Random Forest on the Wine dataset with 9 different
hyperparameter combinations. Each training is recorded as an MLflow
run, with its parameters, accuracy metric and the model itself.
"""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data and split it 80/20. A fixed random_state makes the
# split identical every time, so runs are fairly comparable.
X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Group all runs under one experiment name (created if it doesn't exist).
mlflow.set_experiment("wine-classifier")


def run(n_estimators, max_depth):
    """Train one model and log everything about it as an MLflow run."""
    # Give each run a readable name so it's easy to spot in the UI.
    run_name = f"rf_n{n_estimators}_d{max_depth}"
    with mlflow.start_run(run_name=run_name):
        # Train the model with the given hyperparameters.
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        ).fit(Xtr, ytr)

        # Evaluate on the held-out test set.
        acc = accuracy_score(yte, clf.predict(Xte))

        # Log the inputs (parameters) and the result (metric).
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # Save the trained model as an artifact of this run.
        # Newer MLflow saves sklearn models with "skops", a safe format that
        # refuses tree models unless we say we trust the Tree type. We
        # trained this model ourselves, so it is safe to trust it.
        # input_example lets MLflow record the model's expected input shape.
        mlflow.sklearn.log_model(
            clf,
            name="model",
            input_example=Xtr[:3],
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"n={n_estimators} depth={max_depth} acc={acc:.4f}")


if __name__ == "__main__":
    # Task 3: try every combination -> 3 x 3 = 9 runs.
    for n in (50, 100, 200):
        for d in (3, 5, None):
            run(n, d)