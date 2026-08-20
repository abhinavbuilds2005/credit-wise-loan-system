// CreditWise Application Logic & ML Prediction Engine

document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation Routing
    const navDashboard = document.getElementById("nav-dashboard");
    const navPredict = document.getElementById("nav-predict");
    const dashboardTab = document.getElementById("dashboard-tab");
    const predictTab = document.getElementById("predict-tab");
    const pageTitle = document.getElementById("page-title");
    const pageDescription = document.getElementById("page-description");

    function switchTab(activeBtn, activeTab, titleText, descText) {
        document.querySelectorAll(".nav-btn").forEach(btn => btn.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
        
        activeBtn.classList.add("active");
        activeTab.classList.add("active");
        pageTitle.innerText = titleText;
        pageDescription.innerText = descText;
    }

    navDashboard.addEventListener("click", () => {
        switchTab(
            navDashboard, 
            dashboardTab, 
            "📊 Dashboard Insights", 
            "Interactive metrics, historical distributions, and model performance charts for loan applications."
        );
    });

    navPredict.addEventListener("click", () => {
        switchTab(
            navPredict, 
            predictTab, 
            "🔮 Predict Loan Approval", 
            "Execute live machine learning predictions and risk assessments on unique applicant profiles."
        );
    });

    // 2. Form Sliders Live Sync
    const creditScoreInput = document.getElementById("credit_score");
    const creditScoreVal = document.getElementById("credit_score_val");
    creditScoreInput.addEventListener("input", (e) => {
        const val = e.target.value;
        creditScoreVal.innerText = val;
    });

    const dtiRatioInput = document.getElementById("dti_ratio");
    const dtiRatioVal = document.getElementById("dti_ratio_val");
    dtiRatioInput.addEventListener("input", (e) => {
        const val = Math.round(e.target.value * 100);
        dtiRatioVal.innerText = val + "%";
    });

    // 3. Initialize Dashboard Charts and stats
    initDashboard();

    // 4. Handle Application Form Submission
    const form = document.getElementById("prediction-form");
    form.addEventListener("submit", handleFormSubmit);
});

// Store references to Chart objects to destroy them before re-rendering
let gaugeChartObj = null;
let barChartObj = null;

// Dashboard initialization
function initDashboard() {
    if (typeof DATASET_SUMMARY === "undefined") {
        console.error("DATASET_SUMMARY is not defined. Ensure data.js loaded successfully.");
        return;
    }
    
    const ds = DATASET_SUMMARY;

    // Set KPI figures
    document.getElementById("kpi-total").innerText = ds.total_applicants.toLocaleString();
    document.getElementById("kpi-approved").innerText = ds.approved_applicants.toLocaleString();
    document.getElementById("kpi-rejected").innerText = ds.rejected_applicants.toLocaleString();
    document.getElementById("kpi-rate").innerText = (ds.approval_rate * 100).toFixed(1) + "%";

    // Charts Global Font Configuration
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = "'Inter', sans-serif";

    // Chart 1: Scatter Plot (Credit Score vs Income)
    const ctxScatter = document.getElementById("scatterChart").getContext("2d");
    
    const approvedPoints = ds.scatter_data.filter(p => p.approved === 1).map(p => ({ x: p.credit_score, y: p.income }));
    const rejectedPoints = ds.scatter_data.filter(p => p.approved === 0).map(p => ({ x: p.credit_score, y: p.income }));

    new Chart(ctxScatter, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Approved ✅',
                    data: approvedPoints,
                    backgroundColor: 'rgba(34, 197, 94, 0.65)',
                    borderColor: '#22c55e',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8
                },
                {
                    label: 'Under Review / Rejected ❌',
                    data: rejectedPoints,
                    backgroundColor: 'rgba(239, 68, 68, 0.65)',
                    borderColor: '#ef4444',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Credit Score', color: '#fff', font: { weight: '600' } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    min: 300,
                    max: 850
                },
                y: {
                    title: { display: true, text: 'Applicant Income ($)', color: '#fff', font: { weight: '600' } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                }
            },
            plugins: {
                legend: { position: 'top', labels: { boxWidth: 12, font: { weight: '600' }, color: '#fff' } }
            }
        }
    });

    // Chart 2: Credit Bracket Approval Success
    const ctxCredit = document.getElementById("creditBracketChart").getContext("2d");
    new Chart(ctxCredit, {
        type: 'bar',
        data: {
            labels: ds.credit_bracket_stats.map(d => d.bracket),
            datasets: [{
                label: 'Approval Rate (%)',
                data: ds.credit_bracket_stats.map(d => d.approval_rate * 100),
                backgroundColor: 'rgba(56, 189, 248, 0.75)',
                borderColor: '#38bdf8',
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

    // Chart 3: Loan Purpose Success Rates
    const ctxPurpose = document.getElementById("purposeChart").getContext("2d");
    new Chart(ctxPurpose, {
        type: 'bar',
        data: {
            labels: ds.purpose_stats.map(d => d.purpose),
            datasets: [{
                label: 'Approval Rate (%)',
                data: ds.purpose_stats.map(d => d.approval_rate * 100),
                backgroundColor: 'rgba(129, 140, 248, 0.75)',
                borderColor: '#818cf8',
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

    // Chart 4: Property Area Distribution
    const ctxProperty = document.getElementById("propertyChart").getContext("2d");
    new Chart(ctxProperty, {
        type: 'doughnut',
        data: {
            labels: ds.property_stats.map(d => d.area),
            datasets: [{
                data: ds.property_stats.map(d => d.count),
                backgroundColor: [
                    'rgba(56, 189, 248, 0.75)',  // Semiurban
                    'rgba(245, 158, 11, 0.75)',   // Urban
                    'rgba(239, 68, 68, 0.75)'    // Rural
                ],
                borderColor: '#0f172a',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { boxWidth: 12, color: '#fff' } }
            }
        }
    });
}

// Form prediction handling
function handleFormSubmit(e) {
    e.preventDefault();

    if (typeof MODEL_PARAMETERS === "undefined") {
        alert("ML parameters not loaded. Please ensure data.js exists.");
        return;
    }

    // Get input values
    const income = parseFloat(document.getElementById("applicant_income").value);
    const coIncome = parseFloat(document.getElementById("coapplicant_income").value);
    const loanAmount = parseFloat(document.getElementById("loan_amount").value);
    const loanTerm = parseInt(document.getElementById("loan_term").value);
    const savings = parseFloat(document.getElementById("savings").value);
    const collateral = parseFloat(document.getElementById("collateral").value);
    
    const creditScore = parseInt(document.getElementById("credit_score").value);
    const dti = parseFloat(document.getElementById("dti_ratio").value);
    const existingLoans = parseInt(document.getElementById("existing_loans").value);
    
    const age = parseInt(document.getElementById("age").value);
    const dependents = parseInt(document.getElementById("dependents").value);
    
    const gender = document.getElementById("gender").value;
    const maritalStatus = document.getElementById("marital_status").value;
    const education = document.getElementById("education").value;
    const employmentStatus = document.getElementById("employment_status").value;
    const employerCategory = document.getElementById("employer_category").value;
    const propertyArea = document.getElementById("property_area").value;
    const loanPurpose = document.getElementById("loan_purpose").value;

    // Feature engineering (just like Python script)
    const dtiSq = dti ** 2;
    const creditScoreSq = creditScore ** 2;
    const eduVal = education === "Graduate" ? 0 : 1; // le_edu: Graduate = 0, Not Graduate = 1

    // Build raw feature mapping
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
        "Credit_Score_sq": creditScoreSq
    };

    // One-Hot Encoding simulation: map true/false based on OHE columns
    featDict["Employment_Status_Salaried"] = employmentStatus === "Salaried" ? 1.0 : 0.0;
    featDict["Employment_Status_Self-employed"] = employmentStatus === "Self-employed" ? 1.0 : 0.0;
    featDict["Employment_Status_Unemployed"] = employmentStatus === "Unemployed" ? 1.0 : 0.0;
    
    featDict["Marital_Status_Single"] = maritalStatus === "Single" ? 1.0 : 0.0;
    
    featDict["Loan_Purpose_Car"] = loanPurpose === "Car" ? 1.0 : 0.0;
    featDict["Loan_Purpose_Education"] = loanPurpose === "Education" ? 1.0 : 0.0;
    featDict["Loan_Purpose_Home"] = loanPurpose === "Home" ? 1.0 : 0.0;
    featDict["Loan_Purpose_Personal"] = loanPurpose === "Personal" ? 1.0 : 0.0;
    
    featDict["Property_Area_Semiurban"] = propertyArea === "Semiurban" ? 1.0 : 0.0;
    featDict["Property_Area_Urban"] = propertyArea === "Urban" ? 1.0 : 0.0;
    
    featDict["Gender_Male"] = gender === "Male" ? 1.0 : 0.0;
    
    featDict["Employer_Category_Government"] = employerCategory === "Government" ? 1.0 : 0.0;
    featDict["Employer_Category_MNC"] = employerCategory === "MNC" ? 1.0 : 0.0;
    featDict["Employer_Category_Private"] = employerCategory === "Private" ? 1.0 : 0.0;
    featDict["Employer_Category_Unemployed"] = employerCategory === "Unemployed" ? 1.0 : 0.0;

    // Assemble scaled vector
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

    // Sigmoid probability calculation
    const probability = 1.0 / (1.0 + Math.exp(-z));
    const prediction = probability > 0.5 ? 1 : 0;

    // Set Risk Level
    let riskLevel = "HIGH";
    let riskColorClass = "text-danger";
    if (probability > 0.75) {
        riskLevel = "LOW";
        riskColorClass = "text-success";
    } else if (probability > 0.5) {
        riskLevel = "MEDIUM";
        riskColorClass = "text-warning";
    }

    // Show results section
    const resultsSec = document.getElementById("results-section");
    resultsSec.classList.remove("hidden");

    // Results banner
    const banner = document.getElementById("result-status-banner");
    const bannerTitle = document.getElementById("result-status-title");
    const bannerDesc = document.getElementById("result-status-desc");

    if (prediction === 1) {
        banner.className = "status-banner approved";
        bannerTitle.innerText = "✅ LOAN APPROVED";
        bannerDesc.innerText = "Congratulations! Your application meets our credit approval criteria.";
    } else {
        banner.className = "status-banner review";
        bannerTitle.innerText = "❌ APPLICATION UNDER REVIEW";
        bannerDesc.innerText = "Your application requires further manual risk evaluation.";
    }

    // Populate Metrics Row
    document.getElementById("res-prob").innerText = (probability * 100).toFixed(1) + "%";
    
    const riskBadge = document.getElementById("res-risk");
    riskBadge.innerText = riskLevel;
    riskBadge.className = "res-metric-value " + riskColorClass;
    
    document.getElementById("res-credit").innerText = creditScore;
    document.getElementById("res-dti").innerText = Math.round(dti * 100) + "%";

    // Populate Gauge Chart
    renderGaugeChart(probability);

    // Populate Financial Bar Chart
    renderFinancialChart(income + coIncome, loanAmount, collateral, savings);

    // Populate Decision Intel checklist
    generateChecklists(prediction, creditScore, dti, income, savings, collateral, dependents, loanAmount);

    // Scroll results into view
    resultsSec.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Render semi-doughnut gauge chart using Chart.js
function renderGaugeChart(prob) {
    const ctx = document.getElementById("gaugeChart").getContext("2d");
    
    if (gaugeChartObj) {
        gaugeChartObj.destroy();
    }

    const val = prob * 100;
    let color = "#ef4444"; // high risk red
    if (prob > 0.75) {
        color = "#22c55e"; // low risk green
    } else if (prob > 0.5) {
        color = "#f59e0b"; // med risk yellow
    }

    document.getElementById("gauge-overlay-text").innerText = val.toFixed(1) + "%";

    gaugeChartObj = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Probability', 'Remaining'],
            datasets: [{
                data: [val, 100 - val],
                backgroundColor: [color, '#1e293b'],
                borderWidth: 0
            }]
        },
        options: {
            rotation: -90,
            circumference: 180,
            cutout: '80%',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            }
        }
    });
}

// Render applicant finances summary using Chart.js
function renderFinancialChart(totalIncome, loanAmount, collateral, savings) {
    const ctx = document.getElementById("financialSummaryChart").getContext("2d");

    if (barChartObj) {
        barChartObj.destroy();
    }

    barChartObj = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Total Income', 'Loan Amount', 'Collateral', 'Savings'],
            datasets: [{
                data: [totalIncome, loanAmount, collateral, savings],
                backgroundColor: [
                    'rgba(56, 189, 248, 0.75)',  // Income
                    'rgba(129, 140, 248, 0.75)', // Loan Amount
                    'rgba(34, 197, 94, 0.75)',   // Collateral
                    'rgba(245, 158, 11, 0.75)'   // Savings
                ],
                borderColor: ['#38bdf8', '#818cf8', '#22c55e', '#f59e0b'],
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

// Generate Positive Factors / Risk Considerations dynamically
function generateChecklists(prediction, creditScore, dti, income, savings, collateral, dependents, loanAmount) {
    const title1 = document.getElementById("factors-title-1");
    const list1 = document.getElementById("factors-list-1");
    const title2 = document.getElementById("factors-title-2");
    const list2 = document.getElementById("factors-list-2");

    list1.innerHTML = "";
    list2.innerHTML = "";

    if (prediction === 1) {
        // Approved
        title1.innerText = "✅ Positive Factors";
        title1.className = "factor-header text-success";
        list1.className = "factor-list success-list";

        const positiveFactors = [];
        if (creditScore >= 700) {
            positiveFactors.push(`Excellent Credit Score - <strong>${creditScore}/850</strong>`);
        } else if (creditScore >= 600) {
            positiveFactors.push(`Good Credit Score - <strong>${creditScore}/850</strong>`);
        }
        
        if (dti <= 0.3) {
            positiveFactors.push(`Low Debt-to-Income Ratio - <strong>${Math.round(dti * 100)}%</strong>`);
        } else if (dti <= 0.5) {
            positiveFactors.push(`Moderate Debt-to-Income Ratio - <strong>${Math.round(dti * 100)}%</strong>`);
        }

        if (income >= 50000) {
            positiveFactors.push(`Strong Applicant Income - <strong>$${income.toLocaleString()}</strong>`);
        } else if (income >= 30000) {
            positiveFactors.push(`Stable Applicant Income - <strong>$${income.toLocaleString()}</strong>`);
        }

        if (savings > loanAmount * 0.2) {
            positiveFactors.push(`Good Savings Buffer - <strong>$${savings.toLocaleString()}</strong>`);
        }

        if (collateral >= loanAmount) {
            positiveFactors.push(`Adequate Collateral Coverage - <strong>$${collateral.toLocaleString()}</strong>`);
        }

        if (dependents <= 2) {
            positiveFactors.push(`Low Dependent Load - <strong>${dependents}</strong> dependents`);
        }

        if (positiveFactors.length === 0) {
            positiveFactors.push("Overall strong applicant financial profile");
        }

        positiveFactors.forEach(fact => {
            const li = document.createElement("li");
            li.innerHTML = fact;
            list1.appendChild(li);
        });

        // Risk Considerations
        title2.innerText = "⚠️ Risk Considerations";
        title2.className = "factor-header text-warning";
        list2.className = "factor-list warning-list";

        const considerations = [];
        if (dti > 0.4) {
            considerations.push(`Monitor DTI ratio - Currently high at <strong>${Math.round(dti * 100)}%</strong>`);
        }
        if (creditScore < 750) {
            considerations.push(`Room to improve credit score - Currently <strong>${creditScore}/850</strong>`);
        }
        if (loanAmount > income * 4) {
            considerations.push("High loan-to-income ratio detected");
        }
        if (savings < loanAmount * 0.1) {
            considerations.push("Limited savings buffer compared to loan requested");
        }
        if (dependents > 3) {
            considerations.push(`Multiple dependents (<strong>${dependents}</strong>) may affect monthly repayment capacity`);
        }

        if (considerations.length === 0) {
            considerations.push("No major risk factors identified.");
        }

        considerations.forEach(cons => {
            const li = document.createElement("li");
            li.innerHTML = cons;
            list2.appendChild(li);
        });

    } else {
        // Rejected
        title1.innerText = "⚠️ Key Risk Factors";
        title1.className = "factor-header text-danger";
        list1.className = "factor-list danger-list";

        const riskFactors = [];
        if (dti > 0.4) {
            riskFactors.push(`High DTI Ratio - <strong>${Math.round(dti * 100)}%</strong> (Threshold: 40%)`);
        }
        if (creditScore < 600) {
            riskFactors.push(`Low Credit Score - <strong>${creditScore}/850</strong>`);
        }
        if (income < 30000) {
            riskFactors.push(`Limited Applicant Income - <strong>$${income.toLocaleString()}</strong>`);
        }
        if (savings < loanAmount * 0.1) {
            riskFactors.push(`Insufficient Savings - <strong>$${savings.toLocaleString()}</strong>`);
        }
        if (collateral < loanAmount * 0.5) {
            riskFactors.push(`Low Collateral Coverage - <strong>$${collateral.toLocaleString()}</strong> (Under 50%)`);
        }
        if (dependents > 4) {
            riskFactors.push(`Multiple Dependents - <strong>${dependents}</strong> dependents`);
        }

        if (riskFactors.length === 0) {
            riskFactors.push("Application requires further manual review");
        }

        riskFactors.forEach(risk => {
            const li = document.createElement("li");
            li.innerHTML = risk;
            list1.appendChild(li);
        });

        // Recommended Actions
        title2.innerText = "🔄 Recommended Actions";
        title2.className = "factor-header text-primary";
        list2.className = "factor-list info-list";

        const actions = [];
        if (dti > 0.5) {
            actions.push("<strong>Reduce Debt:</strong> Focus on paying down existing active loans or credit cards.");
        }
        if (loanAmount > income * 3) {
            actions.push("<strong>Lower Loan Amount:</strong> Request a lower principal amount to improve eligibility.");
        }
        if (creditScore < 650) {
            actions.push("<strong>Improve Credit Score:</strong> Build credit score by making timely bill payments.");
        }
        if (collateral < loanAmount) {
            actions.push("<strong>Increase Collateral:</strong> Pledge additional assets to secure the loan and mitigate risk.");
        }
        if (savings < loanAmount * 0.2) {
            actions.push("<strong>Build Savings:</strong> Grow liquid reserves to establish a robust financial cushion.");
        }
        actions.push("<strong>Reapply</strong> after addressing key risk factors.");

        actions.forEach(act => {
            const li = document.createElement("li");
            li.innerHTML = act;
            list2.appendChild(li);
        });
    }
}
