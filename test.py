from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Recreate the exact test split used in training
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # Load model
    model_raw = joblib.load("savedmodel.pth")
    clf = model_raw["model"]

    # Predict on test
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Loaded model test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()