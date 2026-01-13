# UCB-ML-AI-Capstone-Project: Predicting 30-Day Hospital Readmission for Diabetic Patients

## Overview
This capstone project was completed as part of the UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence. The goal is to use machine learning to predict whether a diabetic patient will be readmitted to the hospital within 30 days of discharge and to evaluate how well these models generalize across different real-world datasets.

## Research Question
Can machine learning models accurately predict patient risk and diabetes-related outcomes across different healthcare datasets?

---

## Datasets

### Primary Dataset
**Diabetes 130-US hospitals for years 1999–2008**  
Source: UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
Records: 101,766 patient encounters  

This dataset contains detailed hospital encounter data including demographics, diagnoses, medications, laboratory results, admission types, discharge disposition, and readmission status.

Target variable:
- `readmitted` → converted into `readmitted_binary` (1 = readmitted within 30 days, 0 = otherwise)

---

### External Validation Dataset (Second Dataset)

**Diabetes Prediction Dataset**  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset  
Records: 100,000+ patients  

This dataset contains population-level diabetes risk factors including:
- Age  
- Gender  
- Body Mass Index (BMI)  
- HbA1c level  
- Blood glucose level  
- Smoking history  
- Hypertension  
- Heart disease  

Target variable:
- `diabetes` (1 = diabetic, 0 = non-diabetic)

Although this dataset does not contain hospital readmission labels, it captures core metabolic and demographic features that overlap strongly with the UCI hospital dataset. It is used to test whether the learned feature relationships generalize beyond a single hospital system.

---

## Key Features
- Age, gender, race  
- Number of inpatient, outpatient, and emergency visits  
- Time in hospital  
- Diagnosis codes  
- Diabetes medication and insulin usage  
- HbA1c test results  
- Discharge disposition  

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Examined feature distributions, missing values, and class imbalance  
- Analyzed how readmission rates vary across age, diagnoses, medication use, and hospital utilization  

### 2. Data Preprocessing
- Removed high-missing and non-informative features  
- Encoded categorical variables  
- Applied SMOTE to balance the readmission class  

### 3. Feature Engineering
- Grouped diagnosis codes  
- Created medication change indicators  
- Derived risk-related features from visit history  

### 4. Modeling
Models were trained on the UCI hospital dataset and evaluated on both datasets:

- Logistic Regression  
- XGBoost  

Evaluation metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

The Kaggle dataset was used to validate whether similar features (age, HbA1c, glucose, etc.) remain predictive when applied to an independent population.

---

## Results

The XGBoost model achieved the strongest performance for predicting 30-day hospital readmission on the UCI hospital dataset, with strong recall for identifying high-risk patients. When applied to the Kaggle dataset, the model continued to show meaningful predictive power for diabetes risk, confirming that the learned feature relationships generalize across different data sources.

Key findings:
- Prior inpatient visits, insulin usage, and HbA1c levels were among the most important predictors  
- External validation showed consistent predictive trends across datasets  

---

## Why This Matters

Hospital readmissions are expensive and stressful for patients. By predicting high-risk patients at discharge, healthcare providers can intervene earlier with follow-up care, medication adjustments, or patient education.  

Using two independent datasets strengthens the credibility of this project by demonstrating that the model learns true physiological and behavioral risk patterns, not just hospital-specific data artifacts.

---

## Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn)  
- Jupyter Notebook  
- Matplotlib & Seaborn  
- GitHub  

---

## Next Steps
- Hyperparameter tuning  
- SHAP-based model explainability  
- Deployment via Streamlit for clinical use  
- Testing on additional hospital systems  

---

## Author
Karen Chang  
Licensed RN | MSIT Candidate | UC Berkeley ML/AI Certificate Program
