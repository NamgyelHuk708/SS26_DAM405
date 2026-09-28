from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_split():
    """Load the dataset and return an 80/20 train/test split."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split()
    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)
    print("Train class counts:", {int(c): int((y_train == c).sum()) for c in set(y_train)})