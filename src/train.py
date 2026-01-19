# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # =========================
# # 1. Load Dataset
# # =========================
# df = pd.read_csv("../data/diabetes.csv")

# X = df.drop("Outcome", axis=1)
# y = df["Outcome"]

# # =========================
# # 2. Train-Test Split
# # =========================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # =========================
# # 3. Feature Scaling (IMPORTANT for Gradient Descent)
# # =========================
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # =========================
# # 4. Logistic Regression (Gradient Descent)
# # =========================
# model = LogisticRegression(
#     solver="lbfgs",
#     max_iter=1000
# )

# model.fit(X_train_scaled, y_train)

# # =========================
# # 5. Evaluation
# # =========================
# y_pred = model.predict(X_test_scaled)

# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# # =========================
# # 6. Save Model & Scaler
# # =========================
# joblib.dump(model, "../model/logistic_diabetes_model.joblib")
# joblib.dump(scaler, "../model/scaler.joblib")

# # =========================
# # 7. Output
# # =========================
# print("----- Model Training Report -----")
# print(f"Accuracy: {accuracy:.2%}\n")
# print("Confusion Matrix:")
# print(conf_matrix)
# print("\nClassification Report:")
# print(report)
# print("\nModel and scaler saved successfully.")


import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("data/diabetes.csv")

# ===============================
# 2. Fix Invalid Zero Values
# ===============================
invalid_zero_cols = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)
df[invalid_zero_cols] = df[invalid_zero_cols].fillna(df[invalid_zero_cols].median())

# ===============================
# 3. Feature / Target Split
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ===============================
# 4. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# 5. Scaling (Required)
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 6. Train Logistic Regression
# ===============================
model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# ===============================
# 7. Evaluation
# ===============================
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n----- MODEL EVALUATION -----")
print(f"Accuracy  : {accuracy:.2%}")
print(f"ROC-AUC   : {roc_auc:.3f}\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 8. Save Model & Scaler
# ===============================
joblib.dump(model, "model/logistic_diabetes_model.joblib")
joblib.dump(scaler, "model/scaler.joblib")

print("\nModel and scaler saved successfully.")

