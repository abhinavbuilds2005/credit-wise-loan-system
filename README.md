CrediShield AI | Loan Approval Engine
A production-ready implementation of a binary classification model to predict loan eligibility. This project transitions from a research-oriented Jupyter Notebook to a static client-side web application running predictions directly in the browser via JavaScript.

🛠 Tech Stack
Interface: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript, Chart.js

Modeling: Scikit-Learn (Logistic Regression) serialized to JSON

Data Ops: JavaScript client-side matrix operations

Pre-processing: Robust Imputation, Standard Scaling, and One-Hot Encoding (OHE) implemented client-side in `app.js`

## 🛡️ Architecture & Deployment Robustness
This application includes several enterprise-grade safeguards to guarantee zero server-downtime, zero sleep mode delays, and error-free client-side inference:

1. **Defeats The "Feature Name Mismatch":** The client-side logic extracts `expected_features` directly from the serialized scaler metadata. It mathematically forces the input array into the exact, undeniable order required by the scaler.
2. **"Missing Column" Safety Net (`reindex`):** To avoid the classic One-Hot Encoding missing value crash, the code uses `reindex()` with `fill_value=0`. If a rare combination drops a column, the pipeline intelligently recreates the missing dimension as a `0`.
3. **Replicating Notebook Mathematics Perfectly:** Custom feature engineering runs flawlessly in real-time. By squaring specific risk factors (like DTI ratio and Credit score), the model receives the exact non-linear variables it originally trained around.
4. **Preserving "The Brains":** To avoid raw hardcoding, the app explicitly loads the exact `encoder.pkl` and `edu_encoder.pkl`. Textual categorical inputs transform strictly based on the weights the model actually learned.
5. **Robust Exception Catching:** Prediction logic is wrapped safely with an expanding **Technical Debug Info** screen. Any random edge case won't crash the UI—it will cleanly print the problem dimension for easy maintenance.