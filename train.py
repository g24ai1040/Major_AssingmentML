from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def main():
    # 1. Load data
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data            # shape (400, 4096) -- flattened 64x64 images
    y = data.target          # labels 0..39 (40 persons)

    # 2. Split 70% train / 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # 3. Train DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate on test set
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")

    # 5. Save the model using joblib (assignment requires savedmodel.pth)
    # Use .pth filename per instruction even though it's a sklearn model
    model_path = "savedmodel.pth"
    joblib.dump({"model": clf}, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()