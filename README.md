# UCB-ML-AI-Capstone-Project-Predicting-30-Day-Hospital-Readmission-for-Diabetic-Patients

## Overview
This project is part of the **UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence**.  
It builds machine learning models to predict whether a diabetic patient will be readmitted to the hospital within **30 days after discharge** and evaluates **model robustness through external validation** on an independent diabetes dataset.

The goal is to translate machine learning outputs into **actionable clinical insights** that support discharge planning and reduce preventable hospital readmissions.

---

## Executive Summary (Non-Technical)
Hospital readmissions within 30 days are expensive, stressful for patients, and often preventable.  
This project uses real-world hospital data from over **100,000 patient encounters** to identify diabetic patients who are at high risk of returning to the hospital shortly after discharge.

The models:
- Predict **30-day readmission risk**
- Identify **key drivers of readmission**
- Validate whether these patterns generalize to a **second independent diabetes dataset**

The final system produces interpretable risk scores that can be used by nurses, case managers, and hospital administrators to prioritize follow-up care.

---

## Research Question
Which patient, medication, and hospitalization factors best predict 30-day readmission among diabetic patients, and do these learned patterns generalize to a separate diabetes population?

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

### External Validation Dataset (Second Dataset)
**Name:** Diabetes Prediction Dataset  
**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset  
**Records:** 100,000+ patients  

**Target:** `diabetes` (1 = diabetic, 0 = non-diabetic)

This dataset includes population-level metabolic and lifestyle risk factors such as HbA1c, blood glucose, BMI, smoking history, hypertension, heart disease, age, and gender.  
It is used to test whether learned diabetes risk patterns generalize beyond a single hospital system.

---

## Key Features Used (Primary Dataset)
- Patient demographics (age, gender, race)
- Hospitalization details (time in hospital, number of prior visits)
- Diagnosis codes (diag_1, diag_2, diag_3)
- Diabetes medication usage and insulin status
- Discharge disposition and admission type
- Lab values (A1C result)

---

## Data Cleaning and Feature Engineering
- Removed irrelevant and high-missing columns  
- Encoded categorical variables (Label Encoding and One-Hot Encoding)  
- Standardized numeric features  
- Balanced imbalanced classification targets using **SMOTE**  
- Created multiple modeling targets:
  - `readmitted_binary`
  - `time_in_hospital`
  - `number_inpatient`
  - `A1Cresult`
  - `insulin`

---

## Exploratory Data Analysis (EDA)
EDA was used to explore:
- Readmission rate by age group  
- Length of stay vs. readmission  
- Medication and insulin usage vs. outcomes  
- Lab values and diagnoses vs. outcomes  

All visualizations include readable labels, descriptive titles, legible axes, and appropriate scaling.

---

## Models and Evaluation

**Classification models**
- Logistic Regression (baseline)
- XGBoost (main model)

**Regression models**
- XGBoost Regressor

Models were evaluated using:
- Train / test splits  
- ROC-AUC for classification  
- RMSE for regression  
- Confusion matrices  
- Feature importance  

ROC-AUC was used for classification tasks because the datasets are imbalanced and the goal is to evaluate ranking ability rather than raw accuracy. RMSE was used for regression tasks to measure average prediction error in hospital utilization outcomes.

---

## Results (Primary UCI Dataset)

The models successfully identified patients at high risk for 30-day readmission.

Important drivers included:
- Prior inpatient and emergency visits  
- Discharge disposition  
- Diagnoses  
- Diabetes medications and insulin  
- Lab results (A1C)  

XGBoost provided both higher predictive performance and interpretable feature importance for clinicians.

All plots include labeled axes, descriptive titles, and readable scales for non-technical interpretation.

Outputs are stored in:
```text
├── output/
│ ├── A1Cresult/
│ │ ├── classification_report_A1Cresult.csv
│ │ ├── confusion_matrix_A1Cresult.png
│ │ └── feature_importance_A1Cresult_XGB.png
│ ├── insulin/
│ │ ├── classification_report_insulin.csv
│ │ ├── confusion_matrix_insulin.png
│ │ └── feature_importance_insulin_XGB.png
│ ├── number_inpatient/
│ │ ├── feature_importance_number_inpatient_XGB.png
│ │ └── predictions_number_inpatient.csv
│ ├── readmitted_binary/
│ │ ├── classification_report.csv
│ │ ├── classification_report_readmitted_binary.csv
│ │ ├── confusion_matrix.png
│ │ ├── confusion_matrix_readmitted_binary.png
│ │ ├── feature_importance_readmitted_binary_XGB.png
│ │ ├── roc_readmitted_binary_LogReg.png
│ │ └── roc_readmitted_binary_XGB.png
│ └── time_in_hospital/
│ │ ├── feature_importance_time_in_hospital_XGB.png
│ │ └── predictions_time_in_hospital.csv
```

---

## External Validation Results (Kaggle Dataset)

To evaluate generalization, the same modeling approach was tested on an independent dataset.

| Model | ROC-AUC |
|------|--------|
| Logistic Regression | 0.9625 |
| XGBoost | 0.9800 |

Top predictive features:
- HbA1c_level  
- blood_glucose_level  
- age  
- hypertension  
- heart_disease  

These match known medical risk factors and confirm that the model learned clinically meaningful and transferable patterns.

This reduces the risk that the model is simply memorizing one hospital system and increases confidence in real-world deployment.


Artifacts are stored in:
```text
├── output/
│ ├── external_validation_diabetes/
│ │ ├── classification_report_LogReg_diabetes.csv
│ │ ├── classification_report_XGB_diabetes.csv
│ │ ├── confusion_matrix_LogReg_diabetes.png
│ │ ├── confusion_matrix_XGB_diabetes.png
│ │ ├── feature_importance_XGB_diabetes.png
│ │ ├── feature_importance_XGB_diabetes_top15.csv
│ │ ├── predictions_diabetes_external_validation.csv
│ │ ├── roc_LogReg_diabetes.png
│ │ └── roc_XGB_diabetes.png
```

---

## Business and Clinical Interpretation
The model enables hospitals to:
- Identify high-risk diabetic patients at discharge  
- Prioritize nurse follow-ups and care management  
- Reduce preventable readmissions  
- Improve outcomes while lowering costs  

Without this system, high-risk and low-risk patients are treated the same, increasing avoidable returns.

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
│ ├── A1Cresult/
│ │ ├── classification_report_A1Cresult.csv
│ │ ├── confusion_matrix_A1Cresult.png
│ │ └── feature_importance_A1Cresult_XGB.png
│ ├── external_validation_diabetes/
│ │ ├── classification_report_LogReg_diabetes.csv
│ │ ├── classification_report_XGB_diabetes.csv
│ │ ├── confusion_matrix_LogReg_diabetes.png
│ │ ├── confusion_matrix_XGB_diabetes.png
│ │ ├── feature_importance_XGB_diabetes.png
│ │ ├── feature_importance_XGB_diabetes_top15.csv
│ │ ├── predictions_diabetes_external_validation.csv
│ │ ├── roc_LogReg_diabetes.png
│ │ └── roc_XGB_diabetes.png
│ ├── insulin/
│ │ ├── classification_report_insulin.csv
│ │ ├── confusion_matrix_insulin.png
│ │ └── feature_importance_insulin_XGB.png
│ ├── number_inpatient/
│ │ ├── feature_importance_number_inpatient_XGB.png
│ │ └── predictions_number_inpatient.csv
│ ├── readmitted_binary/
│ │ ├── classification_report.csv
│ │ ├── classification_report_readmitted_binary.csv
│ │ ├── confusion_matrix.png
│ │ ├── confusion_matrix_readmitted_binary.png
│ │ ├── feature_importance_readmitted_binary_XGB.png
│ │ ├── roc_readmitted_binary_LogReg.png
│ │ └── roc_readmitted_binary_XGB.png
│ └── time_in_hospital/
│ │ ├── feature_importance_time_in_hospital_XGB.png
│ │ └── predictions_time_in_hospital.csv
├── Processed Data/
│ └── processed_diabetes_data.csv
├── Capstone_Readmission_Enhanced_Notebook.ipynb
├── INFERENCE.md
├── LICENSE
├── README.md
├── .gitignore
└── requirements.txt
```
All notebooks contain step-by-step comments, modular code, and reproducible pipelines to ensure clarity and maintainability.

---

## Tools Used
- Python (pandas, numpy, scikit-learn, xgboost, imbalanced-learn)
- Jupyter Notebook
- matplotlib, seaborn
- GitHub

---

## Next Steps
- Hyperparameter tuning  
- Add SHAP for explainability  
- Deploy a Streamlit demo  
- Validate on additional healthcare datasets  

---

## Author
Karen Chang  
Licensed RN | MSIT Candidate | UC Berkeley ML/AI Certificate Program
