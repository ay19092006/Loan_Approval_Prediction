import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
from src.predict import load_model, predict_one

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.caption("Educational demo. Predicts the historical approval pattern in the dataset; " "it is not a real credit decision tool.")

@st.cache_resource
def get_model():
    return load_model()
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
with st.form("applicant"):
    c1, c2 = st.columns(2)
    with c1:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        dependents = st.number_input("Number of dependents", 0, 5, 2)
        income = st.number_input("Annual income", 200_000, 9_900_000, 5_000_000, step=100_000)
        loan_amount = st.number_input("Loan amount", 300_000, 39_500_000, 15_000_000, step=100_000)
        term = st.slider("Loan term (years)", 2, 20, 10)
    with c2:
        cibil = st.slider("CIBIL score", 300, 900, 700)
        res = st.number_input("Residential assets", 0, 29_100_000, 5_000_000, step=100_000)
        com = st.number_input("Commercial assets", 0, 19_400_000, 3_000_000, step=100_000)
        lux = st.number_input("Luxury assets", 300_000, 39_200_000, 10_000_000, step=100_000)
        bank = st.number_input("Bank assets", 0, 14_700_000, 3_000_000, step=100_000)
    submitted = st.form_submit_button("Predict", use_container_width=True)
if submitted:
    applicant = {
        "no_of_dependents": dependents, "education": education,
        "self_employed": self_employed, "income_annum": income,
        "loan_amount": loan_amount, "loan_term": term, "cibil_score": cibil,
        "residential_assets_value": res, "commercial_assets_value": com,
        "luxury_assets_value": lux, "bank_asset_value": bank,
    }
    label, p = predict_one(model, applicant)
    if label == "Approved":
        st.success(f"Prediction: **{label}**")
    else:
        st.error(f"Prediction: **{label}**")
    st.progress(p, text=f"Estimated probability of approval: {p:.1%}")
    st.caption(f"Loan-to-income ratio: {loan_amount / income:.2f}")
