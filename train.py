import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import shap

from model import HeartAttackDNN

def preprocess(df, encoders=None, scaler=None, fit=True):
    df = df.drop(columns=["PatientID"], errors="ignore")
    X = df.drop(columns=["HadHeartAttack"], errors="ignore")
    y = df["HadHeartAttack"] if "HadHeartAttack" in df else None
    if fit:
        encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.select_dtypes(include="object")}
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
        return X, y, encoders, scaler
    else:
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
        X = scaler.transform(X)
        return X, y


def train_evaluate(p1="p1.csv", p2="p2.csv"):
    # 1. Load + preprocess
    df_train = pd.read_csv(p1)
    X_train, y_train, enc, scl = preprocess(df_train, fit=True)

    df_test = pd.read_csv(p2)
    X_test, y_test = preprocess(df_test, encoders=enc, scaler=scl, fit=False)

    device = torch.device("cpu")
    model = HeartAttackDNN(input_size=X_train.shape[1]).to(device)

    # 2. DNN Training
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                       torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)),
        batch_size=256, shuffle=True
    )
    for epoch in range(15):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1}/30 – Loss: {total:.4f}")

    # 3. DNN Predictions
    model.eval()
    with torch.no_grad():
        logits_test = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        logits_train = model(torch.tensor(X_train, dtype=torch.float32)).numpy().flatten()
        dnn_preds_test = 1 / (1 + np.exp(-logits_test))
        dnn_preds_train = 1 / (1 + np.exp(-logits_train))

    # 4. Random Forest Training
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    rf_preds_test = rf.predict_proba(X_test)[:, 1]
    rf_preds_train = rf.predict_proba(X_train)[:, 1]

    # 5. Combine using Logistic Regression Meta-Model (stacking)
    stack_train = np.vstack([dnn_preds_train, rf_preds_train]).T
    stack_test = np.vstack([dnn_preds_test, rf_preds_test]).T

    meta_model = LogisticRegression()
    meta_model.fit(stack_train, y_train)
    final_preds = meta_model.predict_proba(stack_test)[:, 1]

    # 6. Evaluation
    threshold = 0.6
    binary_preds = (final_preds >= threshold).astype(int)
    accuracy = (binary_preds == y_test).mean()

    print(f"Precision: {precision_score(y_test, binary_preds):.4f}")
    print(f"Recall:    {recall_score(y_test, binary_preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, binary_preds):.4f}")
    print(f"AUC-ROC:   {roc_auc_score(y_test, final_preds):.4f}")
    print(f"Accuracy:  {accuracy:.2%}")
    print("Meta-model weights (DNN vs RF):", meta_model.coef_, meta_model.intercept_)

    cm = confusion_matrix(y_test, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No HA", "HA"])
    disp.plot(cmap="Blues", values_format='.0f')
    plt.title(f"Confusion Matrix (thr={threshold})")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()  


if __name__ == "__main__":
    train_evaluate()

