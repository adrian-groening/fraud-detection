import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def generate_model():
    # first 10,000 rows of the dataset
    df = pd.read_csv("log.csv", nrows=1000000)

    df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
    df = pd.get_dummies(df, columns=["type"])

    df = df.drop(["nameOrig", "nameDest", "isFlaggedFraud"], axis=1)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    model = xgb.XGBClassifier(scale_pos_weight=round((y == 0).sum() / (y == 1).sum()))
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    model.save_model("fraud.json")

    print(df.shape)
