# Diabetes_Prediction
A production-ready machine learning app that estimates diabetes risk from medical data using probability-based predictions. Designed with healthcare-aware metrics and deployed via Streamlit for practical demonstration.


## 🧠 Tech Stack
- Python
- Scikit-learn (Logistic Regression – Gradient Descent)
- Pandas, NumPy
- Streamlit
- Joblib

## 📊 Dataset
- **PIMA Indians Diabetes Dataset**
- Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

## 🌐 Live Demo
👉 **Streamlit App:**  - https://diabetesprediction-cz2rrkntgnurcahpnqfkfb.streamlit.app/

## ⚙️ How It Works
1. User enters medical details
2. Inputs are scaled using the same preprocessing as training
3. Model outputs a **diabetes risk probability**
4. Risk is categorized as Low / Moderate / High

## 🧪 Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
