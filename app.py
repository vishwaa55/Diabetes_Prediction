import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ===============================
# Load trained model & scaler
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_diabetes_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "This application predicts the **risk of diabetes** based on medical parameters. "
    "The prediction is probability-based and designed to be risk-sensitive."
)

# ===============================
# Input Fields
# ===============================
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=30)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=35)

# ===============================
# Prediction Logic
# ===============================
if st.button("Predict Risk"):
    # Create input dataframe (same column order as training)
    input_data = pd.DataFrame([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # 🔑 Scale input (CRITICAL)
    input_scaled = scaler.transform(input_data)

    # 🔑 Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    # Decision threshold (same as training)
    THRESHOLD = 0.4

    # ===============================
    # Output
    # ===============================
    if probability < 0.3:
        st.success(f"✅ Low Diabetes Risk\n\nProbability: {probability:.2%}")
    elif probability < 0.6:
        st.warning(f"⚠️ Moderate Diabetes Risk\n\nProbability: {probability:.2%}")
    else:
        st.error(f"🚨 High Diabetes Risk\n\nProbability: {probability:.2%}")
