# UCB-ML-AI-Capstone-Project-Predicting-30-Day-Hospital-Readmission-for-Diabetic-Patients

## Overview
This project is part of the **UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence**.  
It builds machine learning models to predict whether a diabetic patient will be readmitted to the hospital within **30 days after discharge** and evaluates **model robustness through cross-validation and external validation**.

The goal is to translate machine learning outputs into **actionable clinical insights** that support discharge planning and reduce preventable hospital readmissions.

---

## Executive Summary (Non-Technical)
Hospital readmissions within 30 days are expensive, stressful for patients, and often preventable.  
This project uses real-world hospital data from over **100,000 patient encounters** to identify diabetic patients who are at high risk of returning to the hospital shortly after discharge.

The models:
- Predict **30-day readmission risk**
- Identify **key drivers of readmission**
- Validate whether learned patterns generalize beyond a single dataset

The final system produces interpretable risk scores that can be used by nurses, case managers, and hospital administrators to prioritize follow-up care.

---

## Research Question
Which patient, medication, and hospitalization factors best predict 30-day readmission among diabetic patients, and do these learned patterns generalize across datasets?

---

## Datasets

### Primary Dataset (Capstone)
**Name:** Diabetes 130-US hospitals for years 1999–2008  
**Source:** UCI Machine Learning Repository  
**Link:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
**Records:** 101,766 hospital encounters  

**Target:**  
`readmitted` → converted to `readmitted_binary`  
(1 = readmitted within 30 days, 0 = otherwise)

---

### External Dataset (Secondary Dataset)
**Name:** Diabetes Prediction Dataset  
**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset  
**Records:** 100,000+ patients  

This dataset contains population-level metabolic and lifestyle features such as HbA1c, blood glucose, BMI, hypertension, heart disease, age, and gender.  
It is used to assess whether diabetes-related risk patterns learned by the models remain clinically meaningful outside a hospital encounter dataset.

---

## Key Features Used (Primary Dataset)
- Patient demographics (age, gender, race)
- Hospital utilization history (time in hospital, number of inpatient and emergency visits)
- Diagnosis codes (diag_1, diag_2, diag_3)
- Diabetes medication usage and insulin status
- Discharge disposition and admission source
- Laboratory results (A1C)

---

## Data Cleaning and Feature Engineering
- Removed irrelevant and high-missing columns  
- Encoded categorical variables using one-hot encoding  
- Standardized numeric features  
- Addressed class imbalance using **SMOTE**  
- Prevented target leakage by removing original readmission labels  

Primary modeling target:
- `readmitted_binary`

Additional exploratory targets were analyzed for supporting insights.

---

## Exploratory Data Analysis (EDA)
EDA was conducted to examine:
- Class imbalance in readmission outcomes  
- Relationships between prior hospital utilization and readmission  
- Effects of diagnoses, medications, and discharge disposition  

All visualizations include human-readable labels, descriptive titles, and appropriate scaling for non-technical interpretation.

---

## Modeling and Evaluation

### Models
- **Logistic Regression** (baseline, interpretable)
- **XGBoost Classifier** (tree-based ensemble)

### Evaluation Strategy
- Stratified train / test split  
- **GridSearchCV with 5-fold cross-validation**
- Primary metric: **ROC-AUC** (appropriate for imbalanced classification)

### Hyperparameter Tuning
Grid search was used to tune:
- Logistic Regression: regularization strength and solver
- XGBoost: number of estimators, learning rate, max depth, subsampling, and column sampling

---

## Cross-Validation Results (5-Fold ROC-AUC)

| Model | CV ROC-AUC |
|------|-----------|
| Logistic Regression (GridSearch) | 0.6506 |
| XGBoost (GridSearch) | 0.6769 |

---

## Held-Out Test Set Results

| Model | Test ROC-AUC |
|------|-------------|
| Logistic Regression (GridSearch) | 0.6601 |
| XGBoost (GridSearch) | 0.6858 |

XGBoost consistently outperformed Logistic Regression, indicating that nonlinear feature interactions play an important role in predicting readmission risk.

---

## Feature Importance (XGBoost)

The most influential predictors of 30-day readmission include:
- `number_inpatient`
- `discharge_disposition_id`
- Diagnosis-related indicators (e.g., V57, V58)
- `number_emergency`
- Diabetes medication changes (e.g., glipizide)

These features align with clinical intuition and highlight actionable opportunities for post-discharge intervention.

---

## External Dataset Analysis (Supplementary Validation)

When applied to the external diabetes dataset, model performance decreased as expected due to dataset differences and target definition changes.  
However, the most influential predictors (glycemic control, comorbidities, age) remained consistent with known clinical risk factors.

This supports the conclusion that the models capture **clinically meaningful diabetes-related risk patterns**, even outside a hospital encounter context.

---

## Business and Clinical Interpretation
This project demonstrates how machine learning can support:
- Early identification of high-risk diabetic patients at discharge  
- Prioritization of nurse follow-ups and care coordination  
- Reduction of preventable readmissions  
- Improved patient outcomes with more efficient resource allocation  

Without such a system, hospitals risk treating high- and low-risk patients identically, leading to avoidable readmissions.

---

## Project Structure
```text
UCB-ML-AI-Capstone-Project-Predicting-30-Day-Hospital-Readmission-for-Diabetic-Patients/
├── documents/
│ └── Required Capstone Assignment 6.1 Draft the Problem Statement.docx
├── notebooks/
│ ├── Capstone Prediction Models.ipynb
│ ├── Capstone_Model_Comparison.ipynb
│ ├── Capstone_Readmission_Enhanced_Notebook.ipynb
│ ├── Capstone_Readmission_Notebook.ipynb
│ ├── Capstone_Readmission_Visuals_Notebook.ipynb
│ └── External_Validation_Kaggle_Diabetes.ipynb
├── Original Data/
│ ├── Diabetes 130-US Hospitals for Years 1999-2008/
│ │ ├── IDS_mapping.csv
│ │ └── diabetic_data.csv
│ ├── Diabetes Prediction Dataset/
│ │ └── diabetes_prediction_dataset.csv
├── output/
│ ├── readmitted_binary/
│ ├── A1Cresult/
│ ├── insulin/
│ ├── number_inpatient/
│ ├── time_in_hospital/
│ └── external_validation_diabetes/
├── Processed Data/
│ └── processed_diabetes_data.csv
├── INFERENCE.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Tools Used
- Python (pandas, numpy, scikit-learn, xgboost, imbalanced-learn)
- Jupyter Notebook
- matplotlib, seaborn
- GitHub

---

## Next Steps
- Add SHAP for enhanced model explainability
- Perform temporal validation on more recent datasets
- Deploy a Streamlit-based clinical risk dashboard

---

## Author
Karen Chang  
Licensed RN | MSIT Candidate | UC Berkeley ML/AI Certificate Program
