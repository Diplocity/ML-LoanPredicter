import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page config
# -----------------------------

st.set_page_config(
    page_title="Loan AI System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load model + encoders
# -----------------------------

model = joblib.load("model.pkl")
label_encoders = joblib.load("encoders.pkl")

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("🏦 Loan AI Dashboard")
st.sidebar.info("This system predicts loan approval using Machine Learning.")

st.sidebar.markdown("### ℹ️ Model Info")
st.sidebar.write("Algorithm: Random Forest")
st.sidebar.write("Type: Classification")

# -----------------------------
# Main UI
# -----------------------------

st.title("🏦 Loan Approval Prediction System")
st.write("AI-powered credit decision assistant")

col1, col2 = st.columns(2)

# -----------------------------
# Inputs
# -----------------------------

with col1:
    st.subheader("👤 Personal Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col2:
    st.subheader("💰 Financial Information")

    applicant_income = st.number_input("Applicant Income", 0)
    coapplicant_income = st.number_input("Coapplicant Income", 0)
    loan_amount = st.number_input("Loan Amount", 0)
    loan_term = st.number_input("Loan Term (months)", 0)

# -----------------------------
# Prediction button
# -----------------------------

st.markdown("---")

predict_btn = st.button("🔍 Predict Loan Approval")

# -----------------------------
# Encode input
# -----------------------------

input_data = pd.DataFrame([{
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "Married": label_encoders["Married"].transform([married])[0],
    "Dependents": label_encoders["Dependents"].transform([dependents])[0],
    "Education": label_encoders["Education"].transform([education])[0],
    "Self_Employed": label_encoders["Self_Employed"].transform([self_employed])[0],
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_history,
    "Property_Area": label_encoders["Property_Area"].transform([property_area])[0]
}])

# -----------------------------
# Prediction + Output
# -----------------------------

if predict_btn:

    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)

    approval = proba[0][1]
    rejection = proba[0][0]

    st.markdown("## 📊 Decision Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### 📈 Risk Analysis")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Approval Probability", f"{approval*100:.2f}%")

    with col4:
        st.metric("Rejection Probability", f"{rejection*100:.2f}%")

    st.progress(float(approval))