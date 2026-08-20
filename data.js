// CreditWise AI Model Parameters and Dataset Summaries
const MODEL_PARAMETERS = {
  "coef": [
    0.47593970734821656,
    0.01389123551392704,
    -0.022648987666231312,
    -0.11436688693384142,
    0.028240511956934083,
    0.04546079875714597,
    -0.009438031520888446,
    -0.2589140211915018,
    -0.1547288050768965,
    -0.16617153217601782,
    -0.25249223162482104,
    -0.15091665357453568,
    -0.25097100361212415,
    0.04333455142722304,
    -0.2604116261909571,
    -0.2129770217220698,
    -0.19723789764498287,
    -0.06811772074563655,
    -0.022506411694100472,
    0.11095022442317974,
    -0.30177337347568667,
    -0.02405494759003113,
    0.19756502562293432,
    0.0979520749737999,
    0.11973846084259071,
    -2.2029713141378884,
    1.8218720503809034
  ],
  "intercept": -1.9762186447342427,
  "scaler": {
    "mean": [
      10910.02554342105,
      5094.864719736842,
      39.989722368421056,
      1.4543302631578945,
      1.9446552631578948,
      9953.86432894737,
      25002.175668421052,
      20440.30707631579,
      47.7,
      0.27375,
      0.5175,
      0.18875,
      0.08875,
      0.33875,
      0.18875,
      0.1875,
      0.1925,
      0.1775,
      0.1875,
      0.52625,
      0.62625,
      0.2025,
      0.14625,
      0.425,
      0.08625,
      0.14082909541551247,
      459630.49335936294
    ],
    "scale": [
      4958.806305634875,
      2868.2235079598686,
      10.929601446859195,
      1.0665994957236982,
      1.3590723673579959,
      5699.333437244647,
      14123.2016080825,
      11174.127878417426,
      23.711389668258587,
      0.4458822013716179,
      0.4996936561534477,
      0.39130989956810447,
      0.2843825548447021,
      0.473284731953187,
      0.3913098995681045,
      0.3903123748998999,
      0.3942635539838802,
      0.382091284904537,
      0.3903123748998999,
      0.49931046203739815,
      0.48379844718642906,
      0.40186284973856445,
      0.3533566717921143,
      0.49434299833212975,
      0.2807328578916263,
      0.09966913728445635,
      93243.28313190119
    ]
  },
  "expected_features": [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Existing_Loans",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "Education_Level",
    "Employment_Status_Salaried",
    "Employment_Status_Self-employed",
    "Employment_Status_Unemployed",
    "Marital_Status_Single",
    "Loan_Purpose_Car",
    "Loan_Purpose_Education",
    "Loan_Purpose_Home",
    "Loan_Purpose_Personal",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
    "Gender_Male",
    "Employer_Category_Government",
    "Employer_Category_MNC",
    "Employer_Category_Private",
    "Employer_Category_Unemployed",
    "DTI_Ratio_sq",
    "Credit_Score_sq"
  ]
};

const DATASET_SUMMARY = {
  "total_applicants": 351,
  "approved_applicants": 113,
  "rejected_applicants": 238,
  "approval_rate": 0.32193732193732194,
  "avg_metrics_by_status": {
    "No": {
      "Credit_Score": 653.609243697479,
      "Applicant_Income": 11037.029411764706,
      "Savings": 9830.642857142857,
      "Collateral_Value": 23823.516806722688,
      "Loan_Amount": 20801.96638655462,
      "DTI_Ratio": 0.4041176470588236,
      "Age": 39.82773109243698
    },
    "Yes": {
      "Credit_Score": 718.8141592920354,
      "Applicant_Income": 12246.991150442478,
      "Savings": 10007.743362831858,
      "Collateral_Value": 23732.814159292036,
      "Loan_Amount": 17269.017699115044,
      "DTI_Ratio": 0.2510619469026549,
      "Age": 38.95575221238938
    }
  },
  "purpose_stats": [
    {
      "purpose": "Personal",
      "count": 64,
      "approval_rate": 0.390625
    },
    {
      "purpose": "Business",
      "count": 75,
      "approval_rate": 0.3466666666666667
    },
    {
      "purpose": "Home",
      "count": 63,
      "approval_rate": 0.2698412698412698
    },
    {
      "purpose": "Education",
      "count": 75,
      "approval_rate": 0.32
    },
    {
      "purpose": "Car",
      "count": 74,
      "approval_rate": 0.28378378378378377
    }
  ],
  "property_stats": [
    {
      "area": "Urban",
      "count": 181,
      "approval_rate": 0.3149171270718232
    },
    {
      "area": "Rural",
      "count": 107,
      "approval_rate": 0.3644859813084112
    },
    {
      "area": "Semiurban",
      "count": 63,
      "approval_rate": 0.2698412698412698
    }
  ],
  "edu_stats": [
    {
      "education": "Not Graduate",
      "count": 106,
      "approval_rate": 0.2830188679245283
    },
    {
      "education": "Graduate",
      "count": 245,
      "approval_rate": 0.33877551020408164
    }
  ],
  "credit_bracket_stats": [
    {
      "bracket": "Excellent (>=750)",
      "count": 63,
      "approval_rate": 0.49206349206349204
    },
    {
      "bracket": "Good (650-749)",
      "count": 149,
      "approval_rate": 0.5503355704697986
    },
    {
      "bracket": "Fair (550-649)",
      "count": 139,
      "approval_rate": 0.0
    },
    {
      "bracket": "Poor (<550)",
      "count": 0,
      "approval_rate": 0
    }
  ],
  "scatter_data": [
    {
      "credit_score": 669.0,
      "income": 11765.0,
      "loan_amount": 22683.0,
      "approved": 1
    },
    {
      "credit_score": 753.0,
      "income": 18448.0,
      "loan_amount": 27179.0,
      "approved": 1
    },
    {
      "credit_score": 746.0,
      "income": 15500.0,
      "loan_amount": 14027.0,
      "approved": 1
    },
    {
      "credit_score": 669.0,
      "income": 10150.0,
      "loan_amount": 11528.0,
      "approved": 1
    },
    {
      "credit_score": 700.0,
      "income": 3678.0,
      "loan_amount": 4996.0,
      "approved": 1
    },
    {
      "credit_score": 770.0,
      "income": 19450.0,
      "loan_amount": 2832.0,
      "approved": 1
    },
    {
      "credit_score": 751.0,
      "income": 7870.0,
      "loan_amount": 16660.0,
      "approved": 1
    },
    {
      "credit_score": 651.0,
      "income": 9043.0,
      "loan_amount": 29080.0,
      "approved": 1
    },
    {
      "credit_score": 669.0,
      "income": 7104.0,
      "loan_amount": 3526.0,
      "approved": 1
    },
    {
      "credit_score": 768.0,
      "income": 18312.0,
      "loan_amount": 7764.0,
      "approved": 1
    },
    {
      "credit_score": 764.0,
      "income": 7959.0,
      "loan_amount": 11402.0,
      "approved": 1
    },
    {
      "credit_score": 700.0,
      "income": 6777.0,
      "loan_amount": 28025.0,
      "approved": 1
    },
    {
      "credit_score": 709.0,
      "income": 12161.0,
      "loan_amount": 8965.0,
      "approved": 1
    },
    {
      "credit_score": 688.0,
      "income": 16596.0,
      "loan_amount": 35329.0,
      "approved": 1
    },
    {
      "credit_score": 692.0,
      "income": 4911.0,
      "loan_amount": 1815.0,
      "approved": 1
    },
    {
      "credit_score": 688.0,
      "income": 18023.0,
      "loan_amount": 10415.0,
      "approved": 1
    },
    {
      "credit_score": 717.0,
      "income": 17147.0,
      "loan_amount": 24490.0,
      "approved": 1
    },
    {
      "credit_score": 750.0,
      "income": 6642.0,
      "loan_amount": 24142.0,
      "approved": 1
    },
    {
      "credit_score": 675.0,
      "income": 16594.0,
      "loan_amount": 12183.0,
      "approved": 1
    },
    {
      "credit_score": 750.0,
      "income": 8731.0,
      "loan_amount": 20465.0,
      "approved": 1
    },
    {
      "credit_score": 708.0,
      "income": 13485.0,
      "loan_amount": 25537.0,
      "approved": 1
    },
    {
      "credit_score": 666.0,
      "income": 11446.0,
      "loan_amount": 37665.0,
      "approved": 1
    },
    {
      "credit_score": 672.0,
      "income": 18157.0,
      "loan_amount": 2614.0,
      "approved": 1
    },
    {
      "credit_score": 785.0,
      "income": 14249.0,
      "loan_amount": 39247.0,
      "approved": 1
    },
    {
      "credit_score": 657.0,
      "income": 14015.0,
      "loan_amount": 22249.0,
      "approved": 1
    },
    {
      "credit_score": 698.0,
      "income": 11106.0,
      "loan_amount": 39680.0,
      "approved": 1
    },
    {
      "credit_score": 751.0,
      "income": 13271.0,
      "loan_amount": 26322.0,
      "approved": 1
    },
    {
      "credit_score": 716.0,
      "income": 12677.0,
      "loan_amount": 2999.0,
      "approved": 1
    },
    {
      "credit_score": 739.0,
      "income": 5913.0,
      "loan_amount": 1730.0,
      "approved": 1
    },
    {
      "credit_score": 693.0,
      "income": 14175.0,
      "loan_amount": 31225.0,
      "approved": 1
    },
    {
      "credit_score": 732.0,
      "income": 19447.0,
      "loan_amount": 32820.0,
      "approved": 1
    },
    {
      "credit_score": 659.0,
      "income": 15949.0,
      "loan_amount": 1947.0,
      "approved": 1
    },
    {
      "credit_score": 734.0,
      "income": 5863.0,
      "loan_amount": 12158.0,
      "approved": 1
    },
    {
      "credit_score": 705.0,
      "income": 12395.0,
      "loan_amount": 39869.0,
      "approved": 1
    },
    {
      "credit_score": 738.0,
      "income": 9574.0,
      "loan_amount": 30489.0,
      "approved": 1
    },
    {
      "credit_score": 796.0,
      "income": 12647.0,
      "loan_amount": 2469.0,
      "approved": 1
    },
    {
      "credit_score": 721.0,
      "income": 17158.0,
      "loan_amount": 10751.0,
      "approved": 1
    },
    {
      "credit_score": 766.0,
      "income": 8949.0,
      "loan_amount": 25474.0,
      "approved": 1
    },
    {
      "credit_score": 708.0,
      "income": 14468.0,
      "loan_amount": 38116.0,
      "approved": 1
    },
    {
      "credit_score": 694.0,
      "income": 13922.0,
      "loan_amount": 8549.0,
      "approved": 1
    },
    {
      "credit_score": 659.0,
      "income": 17265.0,
      "loan_amount": 36690.0,
      "approved": 1
    },
    {
      "credit_score": 779.0,
      "income": 3214.0,
      "loan_amount": 6950.0,
      "approved": 1
    },
    {
      "credit_score": 761.0,
      "income": 18633.0,
      "loan_amount": 14036.0,
      "approved": 1
    },
    {
      "credit_score": 713.0,
      "income": 19019.0,
      "loan_amount": 19660.0,
      "approved": 1
    },
    {
      "credit_score": 735.0,
      "income": 15121.0,
      "loan_amount": 17363.0,
      "approved": 1
    },
    {
      "credit_score": 655.0,
      "income": 13447.0,
      "loan_amount": 15866.0,
      "approved": 1
    },
    {
      "credit_score": 752.0,
      "income": 19589.0,
      "loan_amount": 25281.0,
      "approved": 1
    },
    {
      "credit_score": 745.0,
      "income": 10173.0,
      "loan_amount": 22033.0,
      "approved": 1
    },
    {
      "credit_score": 753.0,
      "income": 7029.0,
      "loan_amount": 9730.0,
      "approved": 1
    },
    {
      "credit_score": 747.0,
      "income": 19868.0,
      "loan_amount": 18993.0,
      "approved": 1
    },
    {
      "credit_score": 777.0,
      "income": 6000.0,
      "loan_amount": 22303.0,
      "approved": 1
    },
    {
      "credit_score": 675.0,
      "income": 6777.0,
      "loan_amount": 7840.0,
      "approved": 1
    },
    {
      "credit_score": 700.0,
      "income": 18984.0,
      "loan_amount": 18131.0,
      "approved": 1
    },
    {
      "credit_score": 675.0,
      "income": 3060.0,
      "loan_amount": 4691.0,
      "approved": 1
    },
    {
      "credit_score": 689.0,
      "income": 10120.0,
      "loan_amount": 39294.0,
      "approved": 1
    },
    {
      "credit_score": 780.0,
      "income": 19586.0,
      "loan_amount": 20589.0,
      "approved": 1
    },
    {
      "credit_score": 718.0,
      "income": 8898.0,
      "loan_amount": 22793.0,
      "approved": 1
    },
    {
      "credit_score": 726.0,
      "income": 11007.0,
      "loan_amount": 6212.0,
      "approved": 1
    },
    {
      "credit_score": 657.0,
      "income": 14383.0,
      "loan_amount": 18997.0,
      "approved": 1
    },
    {
      "credit_score": 734.0,
      "income": 10567.0,
      "loan_amount": 28740.0,
      "approved": 1
    },
    {
      "credit_score": 658.0,
      "income": 6491.0,
      "loan_amount": 14156.0,
      "approved": 1
    },
    {
      "credit_score": 769.0,
      "income": 15808.0,
      "loan_amount": 23578.0,
      "approved": 1
    },
    {
      "credit_score": 735.0,
      "income": 7342.0,
      "loan_amount": 35512.0,
      "approved": 1
    },
    {
      "credit_score": 699.0,
      "income": 7600.0,
      "loan_amount": 1246.0,
      "approved": 1
    },
    {
      "credit_score": 672.0,
      "income": 15417.0,
      "loan_amount": 8904.0,
      "approved": 1
    },
    {
      "credit_score": 720.0,
      "income": 10338.0,
      "loan_amount": 2049.0,
      "approved": 1
    },
    {
      "credit_score": 695.0,
      "income": 9987.0,
      "loan_amount": 2924.0,
      "approved": 1
    },
    {
      "credit_score": 719.0,
      "income": 7373.0,
      "loan_amount": 24813.0,
      "approved": 1
    },
    {
      "credit_score": 797.0,
      "income": 12965.0,
      "loan_amount": 3368.0,
      "approved": 1
    },
    {
      "credit_score": 725.0,
      "income": 6632.0,
      "loan_amount": 10481.0,
      "approved": 1
    },
    {
      "credit_score": 677.0,
      "income": 13473.0,
      "loan_amount": 5887.0,
      "approved": 1
    },
    {
      "credit_score": 662.0,
      "income": 6670.0,
      "loan_amount": 17350.0,
      "approved": 1
    },
    {
      "credit_score": 748.0,
      "income": 15116.0,
      "loan_amount": 20748.0,
      "approved": 1
    },
    {
      "credit_score": 764.0,
      "income": 6729.0,
      "loan_amount": 19704.0,
      "approved": 1
    },
    {
      "credit_score": 729.0,
      "income": 19844.0,
      "loan_amount": 30989.0,
      "approved": 1
    },
    {
      "credit_score": 740.0,
      "income": 15964.0,
      "loan_amount": 22326.0,
      "approved": 0
    },
    {
      "credit_score": 552.0,
      "income": 3021.0,
      "loan_amount": 7021.0,
      "approved": 0
    },
    {
      "credit_score": 646.0,
      "income": 9970.0,
      "loan_amount": 18601.0,
      "approved": 0
    },
    {
      "credit_score": 561.0,
      "income": 7645.0,
      "loan_amount": 27926.0,
      "approved": 0
    },
    {
      "credit_score": 562.0,
      "income": 8420.0,
      "loan_amount": 19170.0,
      "approved": 0
    },
    {
      "credit_score": 603.0,
      "income": 8288.0,
      "loan_amount": 9520.0,
      "approved": 0
    },
    {
      "credit_score": 603.0,
      "income": 5890.0,
      "loan_amount": 29287.0,
      "approved": 0
    },
    {
      "credit_score": 618.0,
      "income": 11692.0,
      "loan_amount": 6681.0,
      "approved": 0
    },
    {
      "credit_score": 661.0,
      "income": 13174.0,
      "loan_amount": 20703.0,
      "approved": 0
    },
    {
      "credit_score": 744.0,
      "income": 6243.0,
      "loan_amount": 34362.0,
      "approved": 0
    },
    {
      "credit_score": 620.0,
      "income": 14977.0,
      "loan_amount": 22066.0,
      "approved": 0
    },
    {
      "credit_score": 739.0,
      "income": 3015.0,
      "loan_amount": 39745.0,
      "approved": 0
    },
    {
      "credit_score": 634.0,
      "income": 2206.0,
      "loan_amount": 7344.0,
      "approved": 0
    },
    {
      "credit_score": 598.0,
      "income": 13246.0,
      "loan_amount": 18769.0,
      "approved": 0
    },
    {
      "credit_score": 615.0,
      "income": 19337.0,
      "loan_amount": 11652.0,
      "approved": 0
    },
    {
      "credit_score": 749.0,
      "income": 4811.0,
      "loan_amount": 27562.0,
      "approved": 0
    },
    {
      "credit_score": 564.0,
      "income": 7258.0,
      "loan_amount": 36670.0,
      "approved": 0
    },
    {
      "credit_score": 719.0,
      "income": 17328.0,
      "loan_amount": 36421.0,
      "approved": 0
    },
    {
      "credit_score": 594.0,
      "income": 9079.0,
      "loan_amount": 39893.0,
      "approved": 0
    },
    {
      "credit_score": 705.0,
      "income": 6297.0,
      "loan_amount": 25918.0,
      "approved": 0
    },
    {
      "credit_score": 602.0,
      "income": 13641.0,
      "loan_amount": 37527.0,
      "approved": 0
    },
    {
      "credit_score": 707.0,
      "income": 19312.0,
      "loan_amount": 22946.0,
      "approved": 0
    },
    {
      "credit_score": 686.0,
      "income": 17444.0,
      "loan_amount": 37527.0,
      "approved": 0
    },
    {
      "credit_score": 704.0,
      "income": 10705.0,
      "loan_amount": 11524.0,
      "approved": 0
    },
    {
      "credit_score": 583.0,
      "income": 12555.0,
      "loan_amount": 32499.0,
      "approved": 0
    },
    {
      "credit_score": 592.0,
      "income": 6944.0,
      "loan_amount": 29715.0,
      "approved": 0
    },
    {
      "credit_score": 555.0,
      "income": 17728.0,
      "loan_amount": 37292.0,
      "approved": 0
    },
    {
      "credit_score": 782.0,
      "income": 2698.0,
      "loan_amount": 26298.0,
      "approved": 0
    },
    {
      "credit_score": 592.0,
      "income": 19364.0,
      "loan_amount": 4338.0,
      "approved": 0
    },
    {
      "credit_score": 752.0,
      "income": 14666.0,
      "loan_amount": 24752.0,
      "approved": 0
    },
    {
      "credit_score": 719.0,
      "income": 17761.0,
      "loan_amount": 7406.0,
      "approved": 0
    },
    {
      "credit_score": 620.0,
      "income": 5490.0,
      "loan_amount": 24940.0,
      "approved": 0
    },
    {
      "credit_score": 584.0,
      "income": 14941.0,
      "loan_amount": 2081.0,
      "approved": 0
    },
    {
      "credit_score": 625.0,
      "income": 14990.0,
      "loan_amount": 25251.0,
      "approved": 0
    },
    {
      "credit_score": 767.0,
      "income": 11146.0,
      "loan_amount": 39515.0,
      "approved": 0
    },
    {
      "credit_score": 629.0,
      "income": 16075.0,
      "loan_amount": 22556.0,
      "approved": 0
    },
    {
      "credit_score": 550.0,
      "income": 11521.0,
      "loan_amount": 18333.0,
      "approved": 0
    },
    {
      "credit_score": 587.0,
      "income": 17343.0,
      "loan_amount": 4192.0,
      "approved": 0
    },
    {
      "credit_score": 760.0,
      "income": 9056.0,
      "loan_amount": 24806.0,
      "approved": 0
    },
    {
      "credit_score": 608.0,
      "income": 14946.0,
      "loan_amount": 4339.0,
      "approved": 0
    },
    {
      "credit_score": 644.0,
      "income": 6495.0,
      "loan_amount": 7931.0,
      "approved": 0
    },
    {
      "credit_score": 577.0,
      "income": 16555.0,
      "loan_amount": 32457.0,
      "approved": 0
    },
    {
      "credit_score": 653.0,
      "income": 12805.0,
      "loan_amount": 9711.0,
      "approved": 0
    },
    {
      "credit_score": 598.0,
      "income": 4849.0,
      "loan_amount": 19932.0,
      "approved": 0
    },
    {
      "credit_score": 780.0,
      "income": 13111.0,
      "loan_amount": 7822.0,
      "approved": 0
    },
    {
      "credit_score": 683.0,
      "income": 15623.0,
      "loan_amount": 18241.0,
      "approved": 0
    },
    {
      "credit_score": 629.0,
      "income": 8776.0,
      "loan_amount": 30764.0,
      "approved": 0
    },
    {
      "credit_score": 759.0,
      "income": 11588.0,
      "loan_amount": 29783.0,
      "approved": 0
    },
    {
      "credit_score": 687.0,
      "income": 11865.0,
      "loan_amount": 9968.0,
      "approved": 0
    },
    {
      "credit_score": 634.0,
      "income": 6873.0,
      "loan_amount": 2218.0,
      "approved": 0
    },
    {
      "credit_score": 628.0,
      "income": 4336.0,
      "loan_amount": 21553.0,
      "approved": 0
    },
    {
      "credit_score": 720.0,
      "income": 8833.0,
      "loan_amount": 30986.0,
      "approved": 0
    },
    {
      "credit_score": 621.0,
      "income": 5054.0,
      "loan_amount": 33813.0,
      "approved": 0
    },
    {
      "credit_score": 791.0,
      "income": 18388.0,
      "loan_amount": 22545.0,
      "approved": 0
    },
    {
      "credit_score": 709.0,
      "income": 12488.0,
      "loan_amount": 24457.0,
      "approved": 0
    },
    {
      "credit_score": 592.0,
      "income": 12636.0,
      "loan_amount": 23889.0,
      "approved": 0
    },
    {
      "credit_score": 597.0,
      "income": 13470.0,
      "loan_amount": 3228.0,
      "approved": 0
    },
    {
      "credit_score": 580.0,
      "income": 3342.0,
      "loan_amount": 32094.0,
      "approved": 0
    },
    {
      "credit_score": 713.0,
      "income": 10096.0,
      "loan_amount": 36017.0,
      "approved": 0
    },
    {
      "credit_score": 792.0,
      "income": 13637.0,
      "loan_amount": 10759.0,
      "approved": 0
    },
    {
      "credit_score": 606.0,
      "income": 8190.0,
      "loan_amount": 14965.0,
      "approved": 0
    },
    {
      "credit_score": 681.0,
      "income": 2504.0,
      "loan_amount": 29906.0,
      "approved": 0
    },
    {
      "credit_score": 629.0,
      "income": 19340.0,
      "loan_amount": 29180.0,
      "approved": 0
    },
    {
      "credit_score": 652.0,
      "income": 18241.0,
      "loan_amount": 23733.0,
      "approved": 0
    },
    {
      "credit_score": 647.0,
      "income": 15446.0,
      "loan_amount": 32541.0,
      "approved": 0
    },
    {
      "credit_score": 788.0,
      "income": 3970.0,
      "loan_amount": 2620.0,
      "approved": 0
    },
    {
      "credit_score": 661.0,
      "income": 14757.0,
      "loan_amount": 28898.0,
      "approved": 0
    },
    {
      "credit_score": 644.0,
      "income": 17305.0,
      "loan_amount": 7615.0,
      "approved": 0
    },
    {
      "credit_score": 604.0,
      "income": 4198.0,
      "loan_amount": 38170.0,
      "approved": 0
    },
    {
      "credit_score": 631.0,
      "income": 13104.0,
      "loan_amount": 1236.0,
      "approved": 0
    },
    {
      "credit_score": 786.0,
      "income": 16397.0,
      "loan_amount": 8896.0,
      "approved": 0
    },
    {
      "credit_score": 777.0,
      "income": 14685.0,
      "loan_amount": 22354.0,
      "approved": 0
    },
    {
      "credit_score": 566.0,
      "income": 7301.0,
      "loan_amount": 3369.0,
      "approved": 0
    },
    {
      "credit_score": 767.0,
      "income": 12998.0,
      "loan_amount": 39278.0,
      "approved": 0
    },
    {
      "credit_score": 743.0,
      "income": 7569.0,
      "loan_amount": 34077.0,
      "approved": 0
    }
  ]
};
