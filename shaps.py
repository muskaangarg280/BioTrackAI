# shap.py

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.drop(columns=["PatientID"], errors="ignore")
    X = df.drop(columns=["HadHeartAttack"], errors="ignore")
    
    if fit:
        encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.select_dtypes(include="object")}
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X, encoders, scaler
    else:
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        X = scaler.transform(X)
        return X

def run_shap():
    # Load the trained Random Forest model (must be saved earlier in train.py)
    rf = joblib.load("random_forest_model.pkl")

    # Load and preprocess data
    train_df = pd.read_csv("p1.csv")
    test_df = pd.read_csv("p2.csv")

    feature_names = train_df.drop(columns=["HadHeartAttack", "PatientID"], errors="ignore").columns.tolist()

    X_train, enc, scl = preprocess(train_df, fit=True)
    X_test = preprocess(test_df, encoders=enc, scaler=scl, fit=False)

    # ---- SHAP for Random Forest ----
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)
    
    # 1. Bar Plot: Feature importance
    shap.summary_plot(
        shap_values[1], X_test,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    
    plt.title("SHAP Feature Importance (Bar)")
    plt.savefig("shap_bar_plot.png", bbox_inches="tight")
    plt.show()  # 👈 Forces it to display
    plt.clf()
    


if __name__ == "__main__":
    run_shap()
