# train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
joblib.dump(model, 'model.joblib')
print('Saved model.joblib')