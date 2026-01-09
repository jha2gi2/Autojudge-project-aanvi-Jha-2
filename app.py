import streamlit as st
import joblib
import os
import re
import numpy as np
from scipy.sparse import hstack

# -----------------------------
# Load models from HOME directory
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def safe_load(path, name):
    if not os.path.exists(path):
        st.error(f"Missing model file: {name}")
        st.stop()
    return joblib.load(path)

tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
svm_clf = joblib.load(os.path.join(MODEL_DIR, "svm_classifier.pkl"))
rf_reg = joblib.load(os.path.join(MODEL_DIR, "rf_regressor.pkl"))

# Load scaler ONLY if you used scaling
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


# -----------------------------
# Feature engineering helpers
# -----------------------------
def count_math_symbols(text):
    return len(re.findall(r"[+\-*/=<>]", text))


def build_features(text):
    tfidf_features = tfidf.transform([text])
    text_len = len(text)
    math_cnt = count_math_symbols(text)

  extra_features = np.array([[text_len, math_cnt]], dtype=np.float64)


    if scaler is not None:
        extra_features = scaler.transform(extra_features)

    return hstack([tfidf_features, extra_features])


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.markdown(
    """
Paste the **problem description**, **input**, and **output**.
The model will predict:
- Difficulty class (Easy / Medium / Hard)
- Numerical difficulty score
"""
)

desc = st.text_area("📄 Problem Description", height=200)
inp = st.text_area("📥 Input Description", height=120)
out = st.text_area("📤 Output Description", height=120)

if st.button("Predict Difficulty"):
    if not desc.strip():
        st.warning("Please enter a problem description.")
    else:
        full_text = desc + " " + inp + " " + out
        X = build_features(full_text)

        difficulty_class = svm_clf.predict(X)[0]
        difficulty_score = rf_reg.predict(X)[0]

        st.success(f"🟢 **Predicted Difficulty Class:** {difficulty_class.capitalize()}")
        st.success(f"📊 **Predicted Difficulty Score:** {difficulty_score:.2f}")
