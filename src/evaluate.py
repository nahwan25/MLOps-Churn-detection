import pandas as pd
import json
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_data():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    return X_test, y_test


def load_model():
    model = load("model/model.joblib")
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
    print("\nAccuracy:", accuracy)

    return accuracy


def save_metrics(accuracy):
    metrics = {
        "accuracy": accuracy
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics disimpan ke metrics.json")


def main():
    X_test, y_test = load_data()
    model = load_model()
    accuracy = evaluate(model, X_test, y_test)
    save_metrics(accuracy)


if __name__ == "__main__":
    main()
