import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from joblib import dump
import os

def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()
    return X_train, y_train


def build_model(model_type="xgb"):
    if model_type == "logreg":
        model = LogisticRegression(max_iter=200)

    elif model_type == "xgb":
        model = Pipeline(steps=[
            ("xgb", XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            ))
        ])
    else:
        raise ValueError("Unsupported model type")

    return model


def main():
    # Load data hasil preprocessing
    X_train, y_train = load_data()

    # Pilih model
    model = build_model(model_type="xgb")

    # Train
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("model", exist_ok=True)
    dump(model, "model/model.joblib")

    print("Training selesai. Model disimpan di model/model.joblib")


if __name__ == "__main__":
    main()
