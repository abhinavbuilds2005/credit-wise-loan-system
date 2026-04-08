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
st.set_page_config(page_title="CreditWise Loan Predictor", layout="wide")
st.title("🏦 CreditWise: Loan Approval Engine")
st.markdown("### Enter applicant details to analyze approval probability.")

# --- 3. INPUT FORM ---
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Financials")
        income = st.number_input("Applicant Income ($)", value=50000)
        co_income = st.number_input("Co-applicant Income ($)", value=20000)
        loan_amount = st.number_input("Loan Amount ($)", value=15000)
        savings = st.number_input("Savings ($)", value=10000)
        collateral = st.number_input("Collateral Value ($)", value=20000)

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
        if prediction == 1:
            st.success(f"🎊 LOAN APPROVED! Confidence: {probability:.2%}")
            st.balloons()
        else:
            st.error(f"❌ LOAN REJECTED. Approval Probability: {probability:.2%}")
        
        # --- 6. DEBUG SECTION (See why it's rejecting) ---
        with st.expander("🔍 View Technical Debug Info"):
            st.write("Raw Probability Score:", probability)
            st.write("Model Type Loaded:", type(model).__name__)
            st.write("Processed Data (First 5 columns):", final_df.iloc[:, :5])
            st.write("Column Count Check:", f"Model expects {len(EXPECTED_FEATURES)}, App sent {final_df.shape[1]}")

    except Exception as e:
        st.error(f"Logic Error: {e}")
        st.info("Ensure the variable names in your Notebook export (joblib.dump) match the inputs here.")