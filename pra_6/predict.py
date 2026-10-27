"""
Task 6: Load the registered model by name and make predictions.

Note that this script never mentions a run ID or file path; the
Model Registry resolves 'latest' to the newest registered version.
"""
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Recreate the same test split used in training.
X, y = load_wine(return_X_y=True)
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the newest version of the registered model.
model = mlflow.sklearn.load_model("models:/wine-classifier/latest")

print("Loaded model prediction:", model.predict(Xte[:5]))
print("Actual labels:          ", yte[:5])