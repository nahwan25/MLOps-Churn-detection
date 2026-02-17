import pandas as pd
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data():
    df = pd.read_csv("data/churn.csv")
    return df


def preprocess(df):
    # Convert TotalCharges ke numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing value jika ada
    df.dropna(inplace=True)

    # Drop ID column
    df.drop('customerID', axis=1, inplace=True)

    # Encode binary categorical
    binary_cols = ['gender', 'Partner', 'Dependents',
                   'PhoneService', 'PaperlessBilling', 'Churn']

    for col in binary_cols:
        df[col] = df[col].map({
            'Yes': 1, 'No': 0,
            'Female': 0, 'Male': 1
        })

    # One-hot encoding multi categorical
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]

    df = pd.get_dummies(df, columns=multi_cat_cols)

    return df


def preprocess_and_split(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print("Distribusi setelah SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())

    return X_train_balanced, y_train_balanced


def save_data(X_train, X_test, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)


def main():
    df = load_data()

    df = preprocess(df)

    X_train, X_test, y_train, y_test = preprocess_and_split(df)

    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    save_data(X_train_balanced, X_test, y_train_balanced, y_test)

    print("Preprocessing selesai. Data disimpan di data/processed/")


if __name__ == "__main__":
    main()
