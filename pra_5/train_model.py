"""
train_model.py  -  Stand-in for the Practical 3 model
------------------------------------------------------
Trains a classifier on the Iris dataset and saves:
  * model.joblib      - the trained model used by app.py
  * train_stats.json  - per-feature mean/min/max of the TRAINING data,
                        used later as the reference for the drift check (Task 5)

If you already have a model from Practical 3, you can keep it, but you
still need train_stats.json for the drift check.
"""
import json
import logging

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from logger_config import get_logger

logger = get_logger(logging.INFO)


def main():
    logger.info("loading Iris dataset")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"training RandomForest on {len(X_train)} rows")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"test accuracy = {acc:.3f}")

    joblib.dump(model, "model.joblib")
    logger.info("model saved to model.joblib")

    # Reference statistics of the training data (used for drift detection)
    stats = {
        "feature_names": [n.replace(" (cm)", "").replace(" ", "_")
                          for n in iris.feature_names],
        "class_names": iris.target_names.tolist(),
        "means": X_train.mean(axis=0).tolist(),
        "stds": X_train.std(axis=0).tolist(),
        "mins": X_train.min(axis=0).tolist(),
        "maxs": X_train.max(axis=0).tolist(),
    }
    with open("train_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("training statistics saved to train_stats.json")


if __name__ == "__main__":
    main()
