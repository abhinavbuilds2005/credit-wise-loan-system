import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. SETUP PATHS & LOAD ASSETS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_asset(file_name):
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path):
        st.error(f"Missing File: {file_name}. Please ensure it is in the same folder as app.py")
        st.stop()
    return joblib.load(path)

# Load the 4 'Brain' components saved from your notebook
try:
    model = load_asset('loan_model.pkl')
    scaler = load_asset('scaler.pkl')
    ohe = load_asset('encoder.pkl')
    le_edu = load_asset('edu_encoder.pkl')
    # Get the exact features and order the scaler expects
    EXPECTED_FEATURES = list(scaler.feature_names_in_)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 2. UI CONFIGURATION ---
st.set_page_config(page_title="CreditWise: AI Loan Risk Intelligence System", layout="wide")

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transform: scale(1.05);
    }
    .metric-card {
        background: linear-gradient(145deg, #1f2428, #24292e);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 5px 5px 10px #080a0f, -5px -5px 10px #121825;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 CreditWise: AI Loan Risk Intelligence System")
st.markdown("### Enter applicant details to analyze approval probability.")

# --- 3. INPUT FORM ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Financials")
        income = st.number_input("Applicant Income ($)", min_value=0, value=50000, step=1000)
        co_income = st.number_input("Co-applicant Income ($)", min_value=0, value=20000, step=1000)
        loan_amount = st.number_input("Loan Amount ($)", min_value=500, value=15000, step=500)
        savings = st.number_input("Savings ($)", min_value=0, value=10000, step=500)
        collateral = st.number_input("Collateral Value ($)", min_value=0, value=20000, step=1000)

    with col2:
        st.subheader("Credit & History")
        credit_score = st.slider("Credit Score", 300, 850, 750)
        dti = st.slider("DTI Ratio", 0.0, 1.0, 0.2)
        loan_term = st.selectbox("Term (Months)", [12, 24, 36, 48, 60, 72, 84])
        dependents = st.number_input("Dependents", 0, 10, 0)
        age = st.number_input("Age", 18, 100, 35)

    with col3:
        st.subheader("Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Married", "Single"])
        employment = st.selectbox("Employment", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        purpose = st.selectbox("Purpose", ["Personal", "Car", "Business", "Home", "Education"])
        employer = st.selectbox("Employer Type", ["Private", "Government", "Unemployed", "MNC", "Business"])

    st.markdown("---")
    submit = st.form_submit_button("🚀 RUN RISK ANALYSIS")

# --- 4. PREDICTION LOGIC ---
if submit:
    try:
        # Create initial DataFrame from inputs
        input_dict = {
            "Applicant_Income": income,
            "Coapplicant_Income": co_income,
            "Age": age,
            "Dependents": dependents,
            "Existing_Loans": 1, 
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "DTI_Ratio": dti,
            "Credit_Score": credit_score,
            "Education_Level": education,
            "Employment_Status": employment,
            "Marital_Status": marital,
            "Loan_Purpose": purpose,
            "Property_Area": property_area,
            "Gender": gender,
            "Employer_Category": employer
        }
        df = pd.DataFrame([input_dict])

        # A. Feature Engineering (The Squares)
        df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
        df["Credit_Score_sq"] = df["Credit_Score"] ** 2
        
        # B. Label Encoding (Education)
        df["Education_Level"] = le_edu.transform(df["Education_Level"])
        
        # C. One-Hot Encoding (Categorical columns)
        ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
        encoded_array = ohe.transform(df[ohe_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
        
        # D. Assembly
        # Combine numerical with encoded, dropping original categoricals and non-squared base columns if needed
        final_df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)
        
        # E. Force EXACT Column Order (Fixes the "Feature Names" Error)
        final_df = final_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
        
        # F. Scaling & Prediction
        scaled_input = scaler.transform(final_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        # --- 5. DISPLAY RESULTS ---
        st.header("Results")
        
        # Risk Classification
        if probability > 0.75:
            risk = "LOW 🟢"
        elif probability > 0.5:
            risk = "MEDIUM 🟡"
        else:
            risk = "HIGH 🔴"

        if prediction == 1:
            st.success("🎊 Loan Approved")
            st.balloons()
            st.markdown(f"**📊 Risk Level:** {risk}")
            st.markdown(f"**📈 Approval Probability:** {probability:.0%}")
            
            reasons = []
            if credit_score >= 700:
                reasons.append("High credit score ✅")
            if dti <= 0.3:
                reasons.append("Low DTI ratio ✅")
            if income >= 50000:
                reasons.append("Strong income ✅")
            if not reasons:
                reasons.append("Overall strong financial profile ✅")
                
            st.markdown("### 📌 Why Approved:")
            for r in reasons:
                st.markdown(f"- {r}")
                
            st.markdown("### 📌 Recommendation:")
            st.markdown("- Offer premium loan plans")
            st.markdown("- Consider cross-selling wealth management products")
        else:
            st.error("❌ Loan Rejected")
            st.markdown(f"**📊 Risk Level:** {risk}")
            st.markdown(f"**📈 Approval Probability:** {probability:.0%}")
            
            reasons = []
            if dti > 0.4:
                reasons.append("High Debt-to-Income ratio ⚠️")
            if credit_score < 600:
                reasons.append("Low credit score ❌")
            if income < 30000:
                reasons.append("Low income ❌")
            if not reasons:
                reasons.append("High overall risk model assessment ⚠️")
                
            st.markdown("### 📌 Why this decision?")
            for r in reasons:
                st.markdown(f"- {r}")
                
            st.markdown("### 📌 Recommended Action:")
            st.markdown("- Reduce loan amount")
            st.markdown("- Increase collateral")
            st.markdown("- Improve credit score")
        
        # --- 6. DEBUG SECTION (See why it's rejecting) ---
        with st.expander("🔍 View Technical Debug Info"):
            st.write("Raw Probability Score:", probability)
            st.write("Model Type Loaded:", type(model).__name__)
            st.write("Processed Data (First 5 columns):", final_df.iloc[:, :5])
            st.write("Column Count Check:", f"Model expects {len(EXPECTED_FEATURES)}, App sent {final_df.shape[1]}")

    except Exception as e:
        st.error(f"Logic Error: {e}")
        st.info("Ensure the variable names in your Notebook export (joblib.dump) match the inputs here.")