import joblib
from sklearn.metrics import accuracy_score

from data import load_split

_, X_test, _, y_test = load_split()

model = joblib.load("automl_model.joblib")
print("Loaded model type:", type(model))
print("Best estimator   :", model.best_estimator)

acc = accuracy_score(y_test, model.predict(X_test))
print("Reloaded model test accuracy:", round(acc, 4))
print("First 5 predictions:", model.predict(X_test[:5]))