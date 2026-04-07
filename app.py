import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="CreditWise - Loan Approval Predictor", layout="centered")

@st.cache_resource
def load_and_train_model():
    # Load dataset
    df = pd.read_csv("loan_approval_data.csv")
    
    # 1. Preprocessing (Same as notebook)
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["number"]).columns
    
    # Impute missing values
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
    
    # Drop ID
    if "Applicant_ID" in df.columns:
        df = df.drop("Applicant_ID", axis=1)
    
    # Label Encoding for Education Level and Target
    le_edu = LabelEncoder()
    df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])
    
    le_target = LabelEncoder()
    df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])
    
    # Feature Engineering (Square DTI and Credit Score)
    df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2
    
    # Drop original columns before OHE to match notebook logic
    X_raw = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]
    
    # One-Hot Encoding
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(X_raw[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=X_raw.index)
    
    X_final = pd.concat([X_raw.drop(columns=ohe_cols), encoded_df], axis=1)
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Train Logistic Regression (Best performing baseline in notebook)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
    return model, scaler, ohe, le_edu, X_final.columns

# Load resources
model, scaler, ohe, le_edu, feature_columns = load_and_train_model()

# --- UI Header ---
st.title("🏦 CreditWise: Loan Approval Predictor")
st.markdown("Enter the applicant details below to check loan eligibility.")

# --- Form Inputs ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0)
        co_income = st.number_input("Co-applicant Income ($)", min_value=0.0, value=0.0)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0.0, value=20000.0)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60, 72, 84])
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Married", "Single"])
    
    with col2:
        age = st.number_input("Age", 18, 100, 30)
        dependents = st.number_input("Number of Dependents", 0, 10, 0)
        dti = st.slider("DTI Ratio (Debt-to-Income)", 0.0, 1.0, 0.3)
        savings = st.number_input("Savings Balance ($)", min_value=0.0, value=1000.0)
        collateral = st.number_input("Collateral Value ($)", min_value=0.0, value=5000.0)
        employment = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    purpose = st.selectbox("Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    employer = st.selectbox("Employer Category", ["Private", "Government", "Unemployed", "MNC", "Business"])

    submit = st.form_submit_button("Predict Approval Status")

# --- Logic ---
if submit:
    # Prepare Input Data
    input_data = pd.DataFrame([{
        "Applicant_Income": income,
        "Coapplicant_Income": co_income,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": 1, # Default value for simulation
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": le_edu.transform([education])[0],
        "Employment_Status": employment,
        "Marital_Status": marital,
        "Loan_Purpose": purpose,
        "Property_Area": property_area,
        "Gender": gender,
        "Employer_Category": employer,
        "DTI_Ratio_sq": dti ** 2,
        "Credit_Score_sq": credit_score ** 2
    }])

    # One-Hot Encode User Input
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    encoded_input = ohe.transform(input_data[ohe_cols])
    encoded_input_df = pd.DataFrame(encoded_input, columns=ohe.get_feature_names_out(ohe_cols), index=input_data.index)
    
    final_input = pd.concat([input_data.drop(columns=ohe_cols), encoded_input_df], axis=1)
    
    # Ensure column order matches training data
    final_input = final_input[feature_columns]
    
    # Scale and Predict
    scaled_input = scaler.transform(final_input)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    # Display Result
    st.divider()
    if prediction == 1:
        st.success(f"🎉 Loan Approved! (Probability: {probability:.2%})")
        st.balloons()
    else:
        st.error(f"❌ Loan Rejected (Approval Probability: {probability:.2%})")