from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data import load_split


def get_baseline():
    """Train the baseline and return (model, test_accuracy)."""
    X_train, X_test, y_train, y_test = load_split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc


if __name__ == "__main__":
    _, base_acc = get_baseline()
    print("Baseline accuracy:", round(base_acc, 4))