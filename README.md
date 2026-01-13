# UCB-ML-AI-Capstone-Project-Predicting-30-Day-Hospital-Readmission-for-Diabetic-Patients

## Overview
This project is part of the UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence. It builds machine learning models to predict whether a diabetic patient will be readmitted to the hospital within 30 days after discharge and demonstrates model robustness through external validation on an independent diabetes dataset.

## Problem Statement
Hospital readmissions within 30 days are costly and often preventable. As a licensed RN in Taiwan, I have seen how difficult it can be to identify high-risk patients at the time of discharge. This project uses machine learning to support early intervention and better discharge planning.

---

## Datasets

### Primary Dataset (Capstone)
**Name**: Diabetes 130-US hospitals for years 1999–2008  
**Source**: UCI Machine Learning Repository  
**Link**: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
**Records**: 101,766 patient encounters  
**Target**: `readmitted` → converted to `readmitted_binary` (`1` = readmitted within 30 days, `0` = otherwise)

### External Validation Dataset (Second Dataset)
**Name**: Diabetes Prediction Dataset  
**Source**: Kaggle  
**Link**: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset  
**Records**: 100,000+ patients  
**Target**: `diabetes` (`1` = diabetic, `0` = non-diabetic)

This second dataset includes population-level metabolic and demographic risk factors such as **age**, **BMI**, **HbA1c**, **blood glucose**, **smoking history**, **hypertension**, and **heart disease**. Although it does not contain hospital readmission labels, it is used to test whether the learned feature relationships generalize beyond a single hospital system.

---

## Key Features Used (Primary Dataset)
- Patient demographics (age, gender, race)
- Hospitalization details (time in hospital, number of prior visits)
- Diagnosis codes (diag_1, diag_2, diag_3)
- Diabetes medication usage and insulin status
- Discharge disposition and admission type
- Lab values (e.g., A1C result)

## Data Preprocessing
- Dropped irrelevant and high-missing columns
- Encoded categorical variables (Label Encoding / One-Hot as needed)
- Balanced the training data for classification tasks using SMOTE (for readmission)

## Models Compared
- Logistic Regression (baseline)
- XGBoost (main model for performance + feature importance)
- (Optional comparisons in earlier iterations: Decision Tree, Random Forest)

---

## Results (Highlights)

### Primary Task: 30-Day Readmission Prediction (UCI)
The project produces standard evaluation artifacts such as:
- Classification report (precision/recall/F1)
- Confusion matrix
- ROC curve (AUC)
- Feature importance (XGBoost)

Outputs are saved under the `output/` folder by target.

### External Validation: Kaggle Diabetes Dataset
To evaluate generalization, the same modeling approach was tested on the Kaggle dataset.

**Performance (Kaggle external validation):**
- Logistic Regression ROC-AUC: **0.9625**
- XGBoost ROC-AUC: **0.9800**

**Top drivers (XGBoost):**
- HbA1c_level
- blood_glucose_level
- age
- hypertension
- heart_disease

This supports that the model captures meaningful diabetes-related risk signals that transfer to an independent dataset.

Artifacts saved in:
`output/external_validation_diabetes/`
- `classification_report_LogReg_diabetes.csv`
- `classification_report_XGB_diabetes.csv`
- `confusion_matrix_LogReg_diabetes.png`
- `confusion_matrix_XGB_diabetes.png`
- `roc_LogReg_diabetes.png`
- `roc_XGB_diabetes.png`
- `feature_importance_XGB_diabetes.png`

---

## Tools Used
- Python (pandas, numpy, scikit-learn, xgboost, imbalanced-learn)
- Jupyter Notebook
- matplotlib, seaborn
- GitHub for version control and documentation

## Next Steps
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Add SHAP for explainable AI insights
- Deploy a simple demo (e.g., Streamlit) for non-technical users
- Validate on additional hospital/system datasets if available

## Author
Karen Chang  
Licensed RN | MSIT Candidate | UC Berkeley ML/AI Certificate Program
