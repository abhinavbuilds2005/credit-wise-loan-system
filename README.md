CreditWise | Loan Approval Engine
A production-ready implementation of a binary classification model to predict loan eligibility. This project transitions from a research-oriented Jupyter Notebook to a deployed Streamlit web interface, utilizing a Scikit-Learn pipeline for real-time inference.

🛠 Tech Stack
Interface: Streamlit

Modeling: Scikit-Learn (Logistic Regression)

Data Ops: Pandas, NumPy

Pre-processing: Robust Imputation, Standard Scaling, and One-Hot Encoding (OHE)

## 🛡️ Architecture & Deployment Robustness
This application includes several enterprise-grade safeguards to guarantee zero downtime and error-free inference against the deployed model:

1. **Defeats The "Feature Name Mismatch":** The Streamlit frontend extracts `EXPECTED_FEATURES` directly from the exact trained `scaler_feature_names_in_`. It mathematically forces the input array into the exact, undeniable order required by the scaler.
2. **"Missing Column" Safety Net (`reindex`):** To avoid the classic One-Hot Encoding missing value crash, the code uses `reindex()` with `fill_value=0`. If a rare combination drops a column, the pipeline intelligently recreates the missing dimension as a `0`.
3. **Replicating Notebook Mathematics Perfectly:** Custom feature engineering runs flawlessly in real-time. By squaring specific risk factors (like DTI ratio and Credit score), the model receives the exact non-linear variables it originally trained around.
4. **Preserving "The Brains":** To avoid raw hardcoding, the app explicitly loads the exact `encoder.pkl` and `edu_encoder.pkl`. Textual categorical inputs transform strictly based on the weights the model actually learned.
5. **Robust Exception Catching:** Prediction logic is wrapped safely with an expanding **Technical Debug Info** screen. Any random edge case won't crash the UI—it will cleanly print the problem dimension for easy maintenance.