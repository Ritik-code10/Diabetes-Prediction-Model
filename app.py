import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")

GOLD, BG, PANEL, TEXT = "#C9A96E", "#0B0B0D", "#141417", "#E8E2D5"
CSV_NAME = "diabetes_prediction_dataset.csv"

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace; }
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: .02em; }
h1 { font-size: 3rem !important; color: #C9A96E; margin-bottom: 0; }
.tagline { color: #8d877a; font-size: .85rem; letter-spacing: .18em; text-transform: uppercase; margin-bottom: 1.5rem; }
div[data-testid="stMetric"] { background: #141417; border: 1px solid #2a2a2e; border-radius: 6px; padding: 14px 18px; }
div[data-testid="stMetricValue"] { color: #C9A96E; font-family: 'Cormorant Garamond', serif; }
.result-card { border-radius: 8px; padding: 28px; text-align: center; margin-top: 12px; border: 1px solid; }
.result-negative { background: linear-gradient(135deg,#141417,#13201a); border-color: #5fae7a; }
.result-positive { background: linear-gradient(135deg,#141417,#241414); border-color: #c96e6e; }
.result-card .label { color: #8d877a; letter-spacing: .2em; font-size: .75rem; text-transform: uppercase; }
.result-card .value { font-family: 'Cormorant Garamond', serif; font-size: 2.6rem; }
.result-negative .value { color: #7fd79c; }
.result-positive .value { color: #e08f8f; }
button[data-baseweb="tab"] { letter-spacing: .08em; }
</style>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update(
    {
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": "#3a3a3f", "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT, "grid.color": "#2a2a2e",
    }
)


# ---------- Data loading (same source as the notebook) ----------
@st.cache_data(show_spinner="Loading dataset from Kaggle…")
def load_from_kaggle():
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    return kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS, "iammustafatz/diabetes-prediction-dataset", CSV_NAME
    )


def get_raw():
    if os.path.exists(CSV_NAME):  # optional: commit the CSV next to app.py
        return pd.read_csv(CSV_NAME)
    try:
        return load_from_kaggle()
    except Exception as e:
        st.warning(f"Could not load the dataset from Kaggle ({type(e).__name__}). Upload the CSV instead.")
        up = st.file_uploader(f"Upload “{CSV_NAME}”", type="csv")
        if up is None:
            st.stop()
        return pd.read_csv(up)


# ---------- Notebook pipeline (Steps 3-6, logic unchanged) ----------
@st.cache_resource(show_spinner="Training model…")
def run_pipeline(raw):
    before_dedup = len(raw)
    df = raw.drop_duplicates().copy()  # Step 3

    encoders = {}  # Step 4 (one LabelEncoder per text column)
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("diabetes", axis=1)
    y = df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)

    return {
        "clean": df, "raw_rows": before_dedup, "encoders": encoders,
        "columns": list(X.columns), "scaler": scaler, "clf": clf,
        "y_test": y_test, "y_pred": y_pred, "metrics": metrics, "cm": cm,
    }


# ---------- App ----------
st.markdown("# Diabetes Prediction")
st.markdown('<div class="tagline">Logistic regression on patient health data</div>', unsafe_allow_html=True)

raw = get_raw()
R = run_pipeline(raw)
clean, encoders = R["clean"], R["encoders"]

tab_pred, tab_model, tab_eda = st.tabs(["Predict", "Model performance", "Data exploration"])

# ----- Predict -----
with tab_pred:
    st.subheader("Check a patient")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", sorted(encoders["gender"].classes_))
    age = c2.number_input("Age", 0, 120, 35)
    smoking = c3.selectbox("Smoking history", sorted(encoders["smoking_history"].classes_))

    c4, c5, c6 = st.columns(3)
    hypertension = c4.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = c5.selectbox("Heart disease", ["No", "Yes"])
    bmi = c6.number_input("BMI", 10.0, 70.0, 27.0, step=0.1)

    c7, c8 = st.columns(2)
    hba1c = c7.number_input("HbA1c level", 3.0, 15.0, 5.8, step=0.1)
    glucose = c8.number_input("Blood glucose level", 50, 400, 120, step=1)

    if st.button("Predict", type="primary"):
        row = {
            "gender": encoders["gender"].transform([gender])[0],
            "age": age,
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "smoking_history": encoders["smoking_history"].transform([smoking])[0],
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glucose,
        }
        x = pd.DataFrame([row])[R["columns"]]
        x_scaled = R["scaler"].transform(x)  # same scaler used for both predict and predict_proba
        pred = R["clf"].predict(x_scaled)[0]
        proba = R["clf"].predict_proba(x_scaled)[0]

        css_class = "result-positive" if pred == 1 else "result-negative"
        label = "Diabetes likely" if pred == 1 else "No diabetes"
        st.markdown(
            f'<div class="result-card {css_class}"><div class="label">Prediction</div>'
            f'<div class="value">{label}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        p1, p2 = st.columns(2)
        p1.metric("No diabetes", f"{proba[0]*100:.1f}%")
        p2.metric("Has diabetes", f"{proba[1]*100:.1f}%")
        st.caption("Logistic Regression with class_weight='balanced'. Not a medical diagnosis.")

# ----- Model performance -----
with tab_model:
    st.subheader("Model performance")
    m = R["metrics"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['Accuracy']:.4f}")
    k2.metric("Precision", f"{m['Precision']:.4f}")
    k3.metric("Recall", f"{m['Recall']:.4f}")
    k4.metric("F1-Score", f"{m['F1-Score']:.4f}")
    st.caption("Recall matters most here — missing an actual diabetic patient is the costly error.")

    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        R["cm"], annot=True, fmt="d", cmap="YlOrBr", ax=ax,
        xticklabels=["No diabetes", "Diabetes"], yticklabels=["No diabetes", "Diabetes"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ----- EDA -----
with tab_eda:
    st.subheader("Data exploration")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows (raw)", f"{R['raw_rows']:,}")
    m2.metric("Rows after de-duplication", f"{len(clean):,}")
    m3.metric("Columns", raw.shape[1])

    st.markdown("**First 5 records**")
    st.dataframe(raw.head())
    st.markdown("**Missing values (raw)**")
    st.dataframe(raw.isnull().sum().rename("missing").to_frame().T)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Target distribution (diabetes)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x="diabetes", data=clean, ax=ax, color=GOLD)
        st.pyplot(fig)
    with e2:
        st.markdown("**Summary statistics**")
        st.dataframe(raw.describe())

    st.markdown("**Correlation heatmap (label-encoded)**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(clean.corr(), annot=True, fmt=".2f", cmap="YlOrBr", ax=ax)
    st.pyplot(fig)
