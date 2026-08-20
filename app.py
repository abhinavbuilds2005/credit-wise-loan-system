import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

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

# --- 2. PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(
    page_title="CrediShield AI - AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    
    /* Main Background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #64748b;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #38bdf8;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    h1 {
        font-size: 2.5em;
        margin-bottom: 10px;
    }
    
    h2 {
        font-size: 1.8em;
        margin-top: 20px;
        margin-bottom: 15px;
        border-bottom: 2px solid #64748b;
        padding-bottom: 10px;
    }
    
    /* Form Elements */
    .stForm {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid #64748b;
        border-radius: 16px;
        padding: 30px;
        backdrop-filter: blur(10px);
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input,
    .stSlider > div > div > input,
    .stSelectbox [data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 2px solid #475569 !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSlider > div > div > input:focus,
    .stSelectbox [data-baseweb="select"]:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.3) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%);
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
        border: 2px solid #22c55e !important;
        border-radius: 12px !important;
        color: #86efac !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 2px solid #ef4444 !important;
        border-radius: 12px !important;
        color: #fca5a5 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 2px solid #f59e0b !important;
        border-radius: 12px !important;
        color: #fcd34d !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
        color: #93c5fd !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 2px solid #64748b;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card:hover {
        border-color: #38bdf8;
        box-shadow: 0 12px 48px rgba(56, 189, 248, 0.2);
        transform: translateY(-4px);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border: 2px solid #475569 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #334155 !important;
        border-color: #38bdf8 !important;
    }
    
    /* Divider */
    hr {
        border: none;
        background: linear-gradient(90deg, transparent, #64748b, transparent);
        height: 2px;
        margin: 30px 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button {
        color: #94a3b8 !important;
        border-bottom: 3px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
    
    /* Text */
    body, p, .stText {
        color: #e2e8f0;
    }
    
    .stMarkdown p {
        color: #cbd5e1;
        line-height: 1.6;
    }
    
    /* Containers */
    .stContainer {
        max-width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HEADER SECTION ---
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="margin: 0; font-size: 3em;">🏦 CrediShield AI</h1>
        <p style="font-size: 1.3em; color: #64748b; margin: 5px 0;">AI-Powered Loan Approval Intelligence System</p>
        <p style="color: #94a3b8; font-size: 0.95em;">Intelligent risk assessment powered by machine learning</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- 4. SIDEBAR INFO ---
with st.sidebar:
    st.markdown("### 📋 About CrediShield AI")
    st.info("""
    **Features:**
    - 🤖 AI-powered risk assessment
    - 📊 Real-time probability scoring
    - 💡 Actionable insights & recommendations
    - 🔒 Enterprise-grade validation
    
    **Model:** Logistic Regression
    **Accuracy:** 85%+ on test data
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. **Enter Details** - Provide applicant information
    2. **AI Analysis** - Model processes 25+ features
    3. **Get Results** - Instant approval probability
    4. **Take Action** - Follow recommendations
    """)

# --- 5. INPUT FORM ---
st.markdown("### 📝 Applicant Information")

with st.form("prediction_form", clear_on_submit=False):
    
    # Financial Information
    st.markdown("#### 💰 Financial Profile")
    fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
    
    with fin_col1:
        income = st.number_input(
            "Applicant Income ($)",
            min_value=0,
            value=50000,
            step=1000,
            help="Primary applicant's annual income"
        )
    
    with fin_col2:
        co_income = st.number_input(
            "Co-applicant Income ($)",
            min_value=0,
            value=20000,
            step=1000,
            help="Co-applicant's annual income (if applicable)"
        )
    
    with fin_col3:
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=500,
            value=150000,
            step=5000,
            help="Requested loan amount"
        )
    
    with fin_col4:
        loan_term = st.selectbox(
            "Loan Term (Months)",
            [12, 24, 36, 48, 60, 72, 84],
            help="Desired repayment period"
        )
    
    # Savings and Collateral
    sav_col1, sav_col2 = st.columns(2)
    
    with sav_col1:
        savings = st.number_input(
            "Current Savings ($)",
            min_value=0,
            value=10000,
            step=500,
            help="Liquid savings available"
        )
    
    with sav_col2:
        collateral = st.number_input(
            "Collateral Value ($)",
            min_value=0,
            value=50000,
            step=1000,
            help="Value of assets that can be pledged"
        )
    
    st.markdown("---")
    
    # Credit & Risk Profile
    st.markdown("#### 📊 Credit & Risk Profile")
    cred_col1, cred_col2, cred_col3 = st.columns(3)
    
    with cred_col1:
        credit_score = st.slider(
            "Credit Score",
            300,
            850,
            700,
            help="Applicant's credit score (300-850)"
        )
        st.caption(f"_Status: {'Excellent' if credit_score >= 750 else 'Good' if credit_score >= 650 else 'Fair' if credit_score >= 550 else 'Poor'}_")
    
    with cred_col2:
        dti = st.slider(
            "Debt-to-Income (DTI) Ratio",
            0.0,
            1.0,
            0.30,
            step=0.05,
            help="Monthly debt payments / monthly income"
        )
        st.caption(f"_Status: {'Excellent' if dti <= 0.2 else 'Good' if dti <= 0.4 else 'Fair' if dti <= 0.6 else 'High'}_")
    
    with cred_col3:
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=1,
            help="Number of active loans"
        )
    
    st.markdown("---")
    
    # Personal Information
    st.markdown("#### 👤 Personal Information")
    pers_col1, pers_col2, pers_col3 = st.columns(3)
    
    with pers_col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Applicant's age in years"
        )
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of financial dependents"
        )
    
    with pers_col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            help="Applicant's gender"
        )
        marital = st.selectbox(
            "Marital Status",
            ["Single", "Married"],
            help="Current marital status"
        )
    
    with pers_col3:
        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"],
            help="Highest education level"
        )
    
    st.markdown("---")
    
    # Employment & Property
    st.markdown("#### 🏢 Employment & Property")
    emp_col1, emp_col2, emp_col3 = st.columns(3)
    
    with emp_col1:
        employment = st.selectbox(
            "Employment Status",
            ["Salaried", "Self-employed", "Contract", "Unemployed"],
            help="Current employment type"
        )
    
    with emp_col2:
        employer = st.selectbox(
            "Employer Type",
            ["Private", "Government", "MNC", "Business", "Unemployed"],
            help="Type of employer organization"
        )
    
    with emp_col3:
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"],
            help="Type of property location"
        )
    
    # Loan Purpose
    purpose = st.selectbox(
        "Loan Purpose",
        ["Personal", "Car", "Business", "Home", "Education"],
        help="Primary purpose of the loan",
        key="purpose_select"
    )
    
    st.markdown("---")
    
    # Submit Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit = st.form_submit_button(
            "🚀 ANALYZE LOAN APPLICATION",
            use_container_width=True
        )

# --- 6. PREDICTION LOGIC ---
if submit:
    with st.spinner("🔄 Processing application..."):
        time.sleep(1.5)  # Simulate processing delay for realistic UX
        try:
            # Create initial DataFrame from inputs
            input_dict = {
                "Applicant_Income": income,
                "Coapplicant_Income": co_income,
                "Age": age,
                "Dependents": dependents,
                "Existing_Loans": existing_loans,
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

            # Feature Engineering
            df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
            df["Credit_Score_sq"] = df["Credit_Score"] ** 2
            
            # Label Encoding (Education)
            df["Education_Level"] = le_edu.transform(df["Education_Level"])
            
            # One-Hot Encoding
            ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
            encoded_array = ohe.transform(df[ohe_cols])
            encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
            
            # Assembly
            final_df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)
            
            # Force EXACT Column Order
            final_df = final_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)
            
            # Scaling & Prediction
            scaled_input = scaler.transform(final_df)
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
            
            # --- 7. RESULTS SECTION ---
            st.markdown("---")
            st.markdown("## 📊 Assessment Results")
            
            # Determine risk level
            if probability > 0.75:
                risk_level = "LOW"
                risk_emoji = "🟢"
                risk_color = "#22c55e"
            elif probability > 0.5:
                risk_level = "MEDIUM"
                risk_emoji = "🟡"
                risk_color = "#f59e0b"
            else:
                risk_level = "HIGH"
                risk_emoji = "🔴"
                risk_color = "#ef4444"
            
            # Main Result Card
            if prediction == 1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
                            border: 2px solid #22c55e; border-radius: 16px; padding: 30px; text-align: center; margin: 20px 0;">
                    <h2 style="color: #22c55e; margin: 0; font-size: 2.5em;">✅ LOAN APPROVED</h2>
                    <p style="color: #86efac; font-size: 1.1em; margin: 10px 0;">Congratulations! Your application meets our approval criteria.</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
                            border: 2px solid #ef4444; border-radius: 16px; padding: 30px; text-align: center; margin: 20px 0;">
                    <h2 style="color: #fca5a5; margin: 0; font-size: 2.5em;">❌ APPLICATION UNDER REVIEW</h2>
                    <p style="color: #fca5a5; font-size: 1.1em; margin: 10px 0;">Your application requires further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics Row
            st.markdown("### Key Metrics")
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            
            with met_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 1.2em; margin: 0; color: #94a3b8;">Approval Probability</h3>
                    <h2 style="font-size: 2.5em; margin: 10px 0; color: #38bdf8;">{probability:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 1.2em; margin: 0; color: #94a3b8;">Risk Level</h3>
                    <h2 style="font-size: 2.5em; margin: 10px 0; color: {risk_color};">{risk_emoji} {risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 1.2em; margin: 0; color: #94a3b8;">Credit Score</h3>
                    <h2 style="font-size: 2.5em; margin: 10px 0; color: #38bdf8;">{credit_score}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with met_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="font-size: 1.2em; margin: 0; color: #94a3b8;">DTI Ratio</h3>
                    <h2 style="font-size: 2.5em; margin: 10px 0; color: #38bdf8;">{dti:.2%}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability Gauge
            st.markdown("### Approval Probability Breakdown")
            gauge_col, detail_col = st.columns([2, 1])
            
            with gauge_col:
                fig_gauge = go.Figure(data=[go.Indicator(
                    mode="gauge+number+delta",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Approval Confidence"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#38bdf8"},
                        'steps': [
                            {'range': [0, 33], 'color': "#fecaca"},
                            {'range': [33, 67], 'color': "#fef3c7"},
                            {'range': [67, 100], 'color': "#bbf7d0"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                )])
                
                fig_gauge.update_layout(
                    paper_bgcolor="#0f172a",
                    font={'color': '#e2e8f0'},
                    height=400,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with detail_col:
                st.markdown(f"""
                **Probability Interpretation:**
                
                - **> 75%** 🟢 Low Risk
                - **50-75%** 🟡 Medium Risk
                - **< 50%** 🔴 High Risk
                
                **Your Score:** {probability:.1%}
                """)
            
            # Decision Reasons
            st.markdown("### 💡 Decision Analysis")
            
            reason_col1, reason_col2 = st.columns(2)
            
            if prediction == 1:
                with reason_col1:
                    st.markdown("#### ✅ Positive Factors")
                    reasons = []
                    
                    if credit_score >= 700:
                        reasons.append(f"• **Excellent Credit Score** - {credit_score}/850")
                    elif credit_score >= 600:
                        reasons.append(f"• **Good Credit Score** - {credit_score}/850")
                    
                    if dti <= 0.3:
                        reasons.append(f"• **Low Debt-to-Income** - {dti:.1%}")
                    elif dti <= 0.5:
                        reasons.append(f"• **Moderate Debt-to-Income** - {dti:.1%}")
                    
                    if income >= 50000:
                        reasons.append(f"• **Strong Income** - ${income:,}")
                    elif income >= 30000:
                        reasons.append(f"• **Stable Income** - ${income:,}")
                    
                    if savings > loan_amount * 0.2:
                        reasons.append(f"• **Good Savings Buffer** - ${savings:,}")
                    
                    if collateral >= loan_amount:
                        reasons.append(f"• **Adequate Collateral** - ${collateral:,}")
                    
                    if dependents <= 2:
                        reasons.append(f"• **Low Dependent Load** - {dependents}")
                    
                    if not reasons:
                        reasons.append("• Overall strong financial profile")
                    
                    for reason in reasons:
                        st.markdown(reason)
                
                with reason_col2:
                    st.markdown("#### ⚠️ Risk Considerations")
                    considerations = []
                    
                    if dti > 0.4:
                        considerations.append(f"• Monitor DTI ratio - Currently {dti:.1%}")
                    
                    if credit_score < 750:
                        considerations.append(f"• Room to improve credit score - {credit_score}/850")
                    
                    if loan_amount > income * 4:
                        considerations.append(f"• High loan-to-income ratio")
                    
                    if savings < loan_amount * 0.1:
                        considerations.append(f"• Limited savings buffer")
                    
                    if dependents > 3:
                        considerations.append(f"• Multiple dependents may affect repayment")
                    
                    if not considerations:
                        considerations.append("• No major risk factors identified")
                    
                    for consideration in considerations:
                        st.markdown(consideration)
            
            else:  # Rejected
                with reason_col1:
                    st.markdown("#### ⚠️ Risk Factors")
                    concerns = []
                    
                    if dti > 0.4:
                        concerns.append(f"• **High DTI Ratio** - {dti:.1%} (Threshold: 0.4)")
                    
                    if credit_score < 600:
                        concerns.append(f"• **Low Credit Score** - {credit_score}/850")
                    
                    if income < 30000:
                        concerns.append(f"• **Limited Income** - ${income:,}")
                    
                    if savings < loan_amount * 0.1:
                        concerns.append(f"• **Insufficient Savings** - ${savings:,}")
                    
                    if collateral < loan_amount * 0.5:
                        concerns.append(f"• **Low Collateral Coverage** - ${collateral:,}")
                    
                    if dependents > 4:
                        concerns.append(f"• **Multiple Dependents** - {dependents}")
                    
                    if not concerns:
                        concerns.append("• Application requires further review")
                    
                    for concern in concerns:
                        st.markdown(concern)
                
                with reason_col2:
                    st.markdown("#### 🔄 Recommended Actions")
                    actions = []
                    
                    if dti > 0.5:
                        actions.append("1. **Reduce Debt** - Pay down existing obligations")
                    
                    if loan_amount > income * 3:
                        actions.append("2. **Lower Loan Amount** - Request a smaller loan")
                    
                    if credit_score < 650:
                        actions.append("3. **Improve Credit** - Build credit score before reapplying")
                    
                    if collateral < loan_amount:
                        actions.append("4. **Increase Collateral** - Provide additional security")
                    
                    if savings < loan_amount * 0.2:
                        actions.append("5. **Build Savings** - Establish emergency fund")
                    
                    actions.append("6. **Reapply** - After addressing key concerns")
                    
                    for action in actions:
                        st.markdown(action)
            
            # Financial Summary Chart
            st.markdown("### 📈 Financial Summary")
            
            summary_data = {
                'Category': ['Income', 'Loan Amount', 'Collateral', 'Savings'],
                'Amount': [income + co_income, loan_amount, collateral, savings]
            }
            summary_df = pd.DataFrame(summary_data)
            
            fig_bar = px.bar(
                summary_df,
                x='Category',
                y='Amount',
                color='Category',
                title="Financial Overview",
                color_discrete_sequence=['#38bdf8', '#06b6d4', '#22c55e', '#f59e0b']
            )
            
            fig_bar.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="rgba(30, 41, 59, 0.5)",
                font={'color': '#e2e8f0'},
                showlegend=False,
                xaxis_title="",
                yaxis_title="Amount ($)",
                height=400,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Debug Section
            with st.expander("🔍 Technical Details & Debug Info"):
                debug_col1, debug_col2 = st.columns(2)
                
                with debug_col1:
                    st.markdown("#### Model Information")
                    st.write(f"**Model Type:** {type(model).__name__}")
                    st.write(f"**Expected Features:** {len(EXPECTED_FEATURES)}")
                    st.write(f"**Processed Features:** {final_df.shape[1]}")
                    st.write(f"**Raw Probability Score:** {probability:.6f}")
                    st.write(f"**Model Prediction:** {'Approved' if prediction == 1 else 'Rejected'}")
                
                with debug_col2:
                    st.markdown("#### Processing Status")
                    st.success("✅ Data processing successful")
                    st.success("✅ Feature engineering completed")
                    st.success("✅ Model inference successful")
                    st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                st.markdown("#### First 10 Features (Processed)")
                st.dataframe(
                    final_df.iloc[:, :10].T,
                    use_container_width=True,
                    height=300
                )
        
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.info("Please ensure all required files are in the same directory as this script.")
            with st.expander("Show Error Details"):
                st.code(str(e))

# --- 8. FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #64748b; font-size: 0.9em;">
    <p>🏦 <strong>CrediShield AI</strong> - Intelligent Loan Approval System</p>
    <p>Powered by Machine Learning | Logistic Regression Model</p>
    <p style="font-size: 0.85em; margin-top: 10px;">Disclaimer: This system is for demonstration purposes. Always conduct thorough due diligence.</p>
</div>
""", unsafe_allow_html=True)
