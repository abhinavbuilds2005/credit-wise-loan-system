// CrediShield AI - Underwriting Intelligence & Prediction Logic

document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation Routing
    const tabs = [
        { btn: document.getElementById("nav-dashboard"), pane: document.getElementById("tab-dashboard"), title: "Dashboard Insights", desc: "Real-time macro portfolio analytics, historical loan distributions, and live risk metrics." },
        { btn: document.getElementById("nav-predict"), pane: document.getElementById("tab-predict"), title: "Live AI Underwriting Predictor", desc: "Execute live machine learning predictions, stress-testing, and dynamic risk explainability." },
        { btn: document.getElementById("nav-portfolio"), pane: document.getElementById("tab-portfolio"), title: "Portfolio Analytics", desc: "Risk-adjusted portfolio segmentation across loan tenures, credit rating distributions, and asset tiers." },
        { btn: document.getElementById("nav-explainer"), pane: document.getElementById("tab-explainer"), title: "Model Explainer & Architecture", desc: "Underwriting engine specifications, validation metrics, and feature sensitivity weights." }
    ];

    const pageTitle = document.getElementById("page-title");
    const pageDescription = document.getElementById("page-description");

    tabs.forEach(tab => {
        if (tab.btn && tab.pane) {
            tab.btn.addEventListener("click", () => {
                tabs.forEach(t => {
                    if (t.btn) t.btn.classList.remove("active");
                    if (t.pane) t.pane.classList.remove("active");
                });
                tab.btn.classList.add("active");
                tab.pane.classList.add("active");
                if (pageTitle) pageTitle.innerText = tab.title;
                if (pageDescription) pageDescription.innerText = tab.desc;
            });
        }
    });

    // 2. Form Sliders Live Sync
    const creditScoreInput = document.getElementById("credit_score");
    const creditScoreVal = document.getElementById("credit_score_val");
    if (creditScoreInput && creditScoreVal) {
        creditScoreInput.addEventListener("input", (e) => {
            creditScoreVal.innerText = e.target.value;
        });
    }

    const dtiRatioInput = document.getElementById("dti_ratio");
    const dtiRatioVal = document.getElementById("dti_ratio_val");
    if (dtiRatioInput && dtiRatioVal) {
        dtiRatioInput.addEventListener("input", (e) => {
            const val = Math.round(e.target.value * 100);
            dtiRatioVal.innerText = val + "%";
        });
    }

    // 3. Prevent mouse wheel from accidentally changing number inputs while scrolling
    document.querySelectorAll('input[type="number"]').forEach(numInput => {
        numInput.addEventListener("wheel", (e) => {
            e.target.blur();
        }, { passive: true });
    });

    // 4. Quick Sample Ingestion Fill Button
    const quickFillBtn = document.getElementById("quick-fill-btn");
    if (quickFillBtn) {
        quickFillBtn.addEventListener("click", () => {
            // Switch to predict tab
            const predictBtn = document.getElementById("nav-predict");
            if (predictBtn) predictBtn.click();

            // Set sample values
            document.getElementById("applicant_income").value = 85000;
            document.getElementById("coapplicant_income").value = 30000;
            document.getElementById("loan_amount").value = 28000;
            document.getElementById("loan_term").value = "36";
            document.getElementById("savings").value = 18000;
            document.getElementById("collateral").value = 45000;
            document.getElementById("credit_score").value = 780;
            document.getElementById("credit_score_val").innerText = "780";
            document.getElementById("dti_ratio").value = 0.22;
            document.getElementById("dti_ratio_val").innerText = "22%";
            document.getElementById("existing_loans").value = 0;
            document.getElementById("age").value = 38;
            document.getElementById("dependents").value = 1;
            document.getElementById("education").value = "Graduate";
            document.getElementById("employment_status").value = "Salaried";
            document.getElementById("employer_category").value = "MNC";
            document.getElementById("property_area").value = "Semiurban";
            document.getElementById("loan_purpose").value = "Home";

            // Trigger prediction
            executePrediction();
        });
    }

    // 5. Initialize Dashboard Charts
    initDashboard();

    // 6. Handle Form Submission
    const form = document.getElementById("prediction-form");
    if (form) {
        form.addEventListener("submit", (e) => {
            e.preventDefault();
            executePrediction();
        });
    }

    // Initial default assessment run on load
    setTimeout(() => {
        executePrediction();
    }, 150);
});

// Chart.js Global Reference
let financialChartObj = null;

// Dashboard initialization
function initDashboard() {
    if (typeof DATASET_SUMMARY === "undefined") {
        console.warn("DATASET_SUMMARY is not defined. Checking data.js...");
        return;
    }
    
    const ds = DATASET_SUMMARY;

    // Set KPI figures
    const kpiTotal = document.getElementById("kpi-total");
    const kpiApproved = document.getElementById("kpi-approved");
    const kpiRejected = document.getElementById("kpi-rejected");
    const kpiRate = document.getElementById("kpi-rate");

    if (kpiTotal) kpiTotal.innerText = ds.total_applicants.toLocaleString();
    if (kpiApproved) kpiApproved.innerText = ds.approved_applicants.toLocaleString();
    if (kpiRejected) kpiRejected.innerText = ds.rejected_applicants.toLocaleString();
    if (kpiRate) kpiRate.innerText = (ds.approval_rate * 100).toFixed(1) + "%";

    // Charts Global Font Configuration
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'Inter', sans-serif";

    // Chart 1: Scatter Plot (Credit Score vs Income)
    const scatterCanvas = document.getElementById("scatterChart");
    if (scatterCanvas) {
        const ctxScatter = scatterCanvas.getContext("2d");
        const approvedPoints = ds.scatter_data.filter(p => p.approved === 1).map(p => ({ x: p.credit_score, y: p.income }));
        const rejectedPoints = ds.scatter_data.filter(p => p.approved === 0).map(p => ({ x: p.credit_score, y: p.income }));

        new Chart(ctxScatter, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Approved ✅',
                        data: approvedPoints,
                        backgroundColor: 'rgba(16, 185, 129, 0.75)',
                        borderColor: '#10b981',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 8
                    },
                    {
                        label: 'Under Review / High Risk ❌',
                        data: rejectedPoints,
                        backgroundColor: 'rgba(244, 63, 94, 0.75)',
                        borderColor: '#f43f5e',
                        borderWidth: 1,
                        pointRadius: 5,
                        pointHoverRadius: 8
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: { display: true, text: 'Credit Score', color: '#cbd5e1', font: { weight: '600' } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        min: 300,
                        max: 850
                    },
                    y: {
                        title: { display: true, text: 'Applicant Income ($)', color: '#cbd5e1', font: { weight: '600' } },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' }
                    }
                },
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12, font: { weight: '600' }, color: '#e2e8f0' } }
                }
            }
        });
    }

    // Chart 2: Credit Bracket Approval Success
    const creditCanvas = document.getElementById("creditBracketChart");
    if (creditCanvas) {
        const ctxCredit = creditCanvas.getContext("2d");
        new Chart(ctxCredit, {
            type: 'bar',
            data: {
                labels: ds.credit_bracket_stats.map(d => d.bracket),
                datasets: [{
                    label: 'Approval Rate (%)',
                    data: ds.credit_bracket_stats.map(d => (d.approval_rate * 100).toFixed(1)),
                    backgroundColor: 'rgba(0, 240, 255, 0.75)',
                    borderColor: '#00f0ff',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { min: 0, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    x: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // Chart 3: Loan Purpose Success Rates
    const purposeCanvas = document.getElementById("purposeChart");
    if (purposeCanvas) {
        const ctxPurpose = purposeCanvas.getContext("2d");
        new Chart(ctxPurpose, {
            type: 'bar',
            data: {
                labels: ds.purpose_stats.map(d => d.purpose),
                datasets: [{
                    label: 'Approval Rate (%)',
                    data: ds.purpose_stats.map(d => (d.approval_rate * 100).toFixed(1)),
                    backgroundColor: 'rgba(99, 102, 241, 0.75)',
                    borderColor: '#6366f1',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {
                    x: { min: 0, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                    y: { grid: { display: false } }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // Chart 4: Property Area Distribution
    const propertyCanvas = document.getElementById("propertyChart");
    if (propertyCanvas) {
        const ctxProperty = propertyCanvas.getContext("2d");
        new Chart(ctxProperty, {
            type: 'doughnut',
            data: {
                labels: ds.property_stats.map(d => d.area),
                datasets: [{
                    data: ds.property_stats.map(d => d.count),
                    backgroundColor: [
                        'rgba(0, 240, 255, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(244, 63, 94, 0.8)'
                    ],
                    borderColor: '#050d1a',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { boxWidth: 12, color: '#e2e8f0' } }
                }
            }
        });
    }
}

// Prediction Execution Logic
function executePrediction() {
    if (typeof MODEL_PARAMETERS === "undefined") {
        console.error("ML parameters not loaded. Please ensure data.js exists.");
        return;
    }

    // Get input values safely
    const income = parseFloat(document.getElementById("applicant_income")?.value || 50000);
    const coIncome = parseFloat(document.getElementById("coapplicant_income")?.value || 0);
    const loanAmount = parseFloat(document.getElementById("loan_amount")?.value || 25000);
    const loanTerm = parseInt(document.getElementById("loan_term")?.value || 84);
    const savings = parseFloat(document.getElementById("savings")?.value || 10000);
    const collateral = parseFloat(document.getElementById("collateral")?.value || 30000);
    
    const creditScore = parseInt(document.getElementById("credit_score")?.value || 700);
    const dti = parseFloat(document.getElementById("dti_ratio")?.value || 0.3);
    const existingLoans = parseInt(document.getElementById("existing_loans")?.value || 1);
    
    const age = parseInt(document.getElementById("age")?.value || 35);
    const dependents = parseInt(document.getElementById("dependents")?.value || 0);
    
    const gender = document.getElementById("gender")?.value || "Male";
    const maritalStatus = document.getElementById("marital_status")?.value || "Single";
    const education = document.getElementById("education")?.value || "Graduate";
    const employmentStatus = document.getElementById("employment_status")?.value || "Salaried";
    const employerCategory = document.getElementById("employer_category")?.value || "Private";
    const propertyArea = document.getElementById("property_area")?.value || "Semiurban";
    const loanPurpose = document.getElementById("loan_purpose")?.value || "Personal";

    // Feature engineering
    const dtiSq = dti ** 2;
    const creditScoreSq = creditScore ** 2;
    const eduVal = education === "Graduate" ? 0 : 1;

    // Feature dictionary
    const featDict = {
        "Applicant_Income": income,
        "Coapplicant_Income": coIncome,
        "Age": age,
        "Dependents": dependents,
        "Existing_Loans": existingLoans,
        "Savings": savings,
        "Collateral_Value": collateral,
        "Loan_Amount": loanAmount,
        "Loan_Term": loanTerm,
        "Education_Level": eduVal,
        "DTI_Ratio_sq": dtiSq,
        "Credit_Score_sq": creditScoreSq,
        "Employment_Status_Salaried": employmentStatus === "Salaried" ? 1.0 : 0.0,
        "Employment_Status_Self-employed": employmentStatus === "Self-employed" ? 1.0 : 0.0,
        "Employment_Status_Unemployed": employmentStatus === "Unemployed" ? 1.0 : 0.0,
        "Marital_Status_Single": maritalStatus === "Single" ? 1.0 : 0.0,
        "Loan_Purpose_Car": loanPurpose === "Car" ? 1.0 : 0.0,
        "Loan_Purpose_Education": loanPurpose === "Education" ? 1.0 : 0.0,
        "Loan_Purpose_Home": loanPurpose === "Home" ? 1.0 : 0.0,
        "Loan_Purpose_Personal": loanPurpose === "Personal" ? 1.0 : 0.0,
        "Property_Area_Semiurban": propertyArea === "Semiurban" ? 1.0 : 0.0,
        "Property_Area_Urban": propertyArea === "Urban" ? 1.0 : 0.0,
        "Gender_Male": gender === "Male" ? 1.0 : 0.0,
        "Employer_Category_Government": employerCategory === "Government" ? 1.0 : 0.0,
        "Employer_Category_MNC": employerCategory === "MNC" ? 1.0 : 0.0,
        "Employer_Category_Private": employerCategory === "Private" ? 1.0 : 0.0,
        "Employer_Category_Unemployed": employerCategory === "Unemployed" ? 1.0 : 0.0
    };

    // Scaled vector calculation
    const expected = MODEL_PARAMETERS.expected_features;
    const mean = MODEL_PARAMETERS.scaler.mean;
    const scale = MODEL_PARAMETERS.scaler.scale;
    const coef = MODEL_PARAMETERS.coef;
    const intercept = MODEL_PARAMETERS.intercept;

    let z = intercept;
    for (let i = 0; i < expected.length; i++) {
        const val = featDict[expected[i]] || 0.0;
        const scaledVal = (val - mean[i]) / scale[i];
        z += scaledVal * coef[i];
    }

    // Sigmoid probability
    const probability = 1.0 / (1.0 + Math.exp(-z));
    const isApproved = probability >= 0.5;

    // Update UI Circular Meter
    const meterPercent = document.getElementById("meter-percent-val");
    const meterStroke = document.getElementById("meter-stroke");
    const decisionBadge = document.getElementById("decision-badge");
    const decisionText = document.getElementById("decision-text");
    const decisionIcon = document.getElementById("decision-icon");

    const pctNumber = (probability * 100).toFixed(1);
    if (meterPercent) meterPercent.innerText = pctNumber + "%";

    // Circle Circumference for r=65 is 2 * PI * 65 ≈ 408.4
    const circumference = 408.4;
    const offset = circumference - (circumference * probability);
    if (meterStroke) {
        meterStroke.style.strokeDasharray = `${circumference}`;
        meterStroke.style.strokeDashoffset = `${offset}`;
        if (probability >= 0.75) {
            meterStroke.style.stroke = "#00f0ff";
            meterStroke.style.filter = "drop-shadow(0 0 10px rgba(0,240,255,0.6))";
        } else if (probability >= 0.5) {
            meterStroke.style.stroke = "#f59e0b";
            meterStroke.style.filter = "drop-shadow(0 0 10px rgba(245,158,11,0.6))";
        } else {
            meterStroke.style.stroke = "#f43f5e";
            meterStroke.style.filter = "drop-shadow(0 0 10px rgba(244,63,94,0.6))";
        }
    }

    // Update Decision Badge
    if (decisionBadge) {
        if (probability >= 0.75) {
            decisionBadge.className = "decision-badge approved";
            decisionText.innerText = "APPROVED - Low Risk";
            decisionIcon.innerText = "verified";
        } else if (probability >= 0.5) {
            decisionBadge.className = "decision-badge medium";
            decisionText.innerText = "CONDITIONAL APPROVAL";
            decisionIcon.innerText = "pending";
        } else {
            decisionBadge.className = "decision-badge review";
            decisionText.innerText = "UNDER REVIEW - High Risk";
            decisionIcon.innerText = "cancel";
        }
    }

    // Update Factors (SHAP Drivers)
    const factorList1 = document.getElementById("factors-list-1");
    if (factorList1) {
        factorList1.innerHTML = "";
        
        // Dynamic influence calculation
        const creditImpact = (creditScore >= 700) ? `+${((creditScore - 600) / 10).toFixed(1)}%` : `-${((700 - creditScore) / 10).toFixed(1)}%`;
        const creditIsPos = creditScore >= 650;

        const collateralRatio = (collateral / Math.max(loanAmount, 1));
        const collateralImpact = collateralRatio >= 1 ? `+${(collateralRatio * 10).toFixed(1)}%` : `-${((1 - collateralRatio) * 15).toFixed(1)}%`;
        const collateralIsPos = collateralRatio >= 0.8;

        const dtiImpact = dti <= 0.35 ? `+${((0.4 - dti) * 30).toFixed(1)}%` : `-${((dti - 0.35) * 45).toFixed(1)}%`;
        const dtiIsPos = dti <= 0.35;

        const factors = [
            { name: "Credit Score Trajectory", icon: "analytics", val: creditImpact, isPos: creditIsPos },
            { name: "Collateral Coverage", icon: "shield", val: collateralImpact, isPos: collateralIsPos },
            { name: "DTI Debt Leverage", icon: "trending_down", val: dtiImpact, isPos: dtiIsPos }
        ];

        factors.forEach(f => {
            const li = document.createElement("li");
            li.innerHTML = `
                <span class="factor-name"><span class="material-symbols-outlined" style="font-size: 16px; color: ${f.isPos ? 'var(--secondary-emerald)' : 'var(--danger-rose)'};">${f.icon}</span> ${f.name}</span>
                <span class="factor-val ${f.isPos ? 'pos' : 'neg'}">${f.val}</span>
            `;
            factorList1.appendChild(li);
        });
    }

    // Update Narrative Summary
    const underwriterNarrative = document.getElementById("underwriter-narrative");
    const factorList2 = document.getElementById("factors-list-2");
    if (underwriterNarrative && factorList2) {
        factorList2.innerHTML = "";
        if (isApproved) {
            underwriterNarrative.innerText = `Applicant demonstrates exceptional liquidity and creditworthiness with a projected approval confidence of ${pctNumber}%.`;
            const tips = [
                `<strong>Leverage:</strong> Total collateral coverage is ${(collateral / loanAmount).toFixed(1)}x requested principal.`,
                `<strong>Max Recommended:</strong> Up to $${Math.round(income * 0.6).toLocaleString()} based on debt threshold.`
            ];
            tips.forEach(t => {
                const li = document.createElement("li");
                li.style.fontSize = "12px";
                li.innerHTML = t;
                factorList2.appendChild(li);
            });
        } else {
            underwriterNarrative.innerText = `Applicant profile requires enhanced underwriting scrutiny. Projected confidence is ${pctNumber}% due to risk concentration.`;
            const tips = [
                `<strong>Action:</strong> Lower loan principal or increase asset pledges to reduce debt burden.`,
                `<strong>DTI Ratio:</strong> Target DTI ratio below 35% (currently ${Math.round(dti * 100)}%).`
            ];
            tips.forEach(t => {
                const li = document.createElement("li");
                li.style.fontSize = "12px";
                li.innerHTML = t;
                factorList2.appendChild(li);
            });
        }
    }

    // Render Financial Balance Chart
    renderFinancialChart(income + coIncome, loanAmount, collateral, savings);
}

// Financial capacity summary bar chart
function renderFinancialChart(totalIncome, loanAmount, collateral, savings) {
    const canvas = document.getElementById("financialSummaryChart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (financialChartObj) {
        financialChartObj.destroy();
    }

    financialChartObj = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Total Income', 'Loan Amount', 'Collateral', 'Savings'],
            datasets: [{
                data: [totalIncome, loanAmount, collateral, savings],
                backgroundColor: [
                    'rgba(0, 240, 255, 0.75)',
                    'rgba(99, 102, 241, 0.75)',
                    'rgba(16, 185, 129, 0.75)',
                    'rgba(245, 158, 11, 0.75)'
                ],
                borderColor: ['#00f0ff', '#6366f1', '#10b981', '#f59e0b'],
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } },
                x: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
