# UCB-ML-AI-Capstone-Project-Predicting-30-Day-Hospital-Readmission-for-Diabetic-Patients

## Overview
This project is part of the UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence. It aims to build a machine learning model to predict whether a diabetic patient will be readmitted to the hospital within 30 days after discharge.

## Problem Statement
Hospital readmissions within 30 days are costly and often preventable. As a nurse with clinical experience, I have seen the challenges in identifying high-risk patients at the time of discharge. This project uses machine learning to assist care teams in early intervention.

## Dataset
**Name**: Diabetes 130-US hospitals for years 1999–2008  
**Source**: UCI Machine Learning Repository  
**Records**: 101,766 patient encounters  
**Target**: `readmitted` → Converted to binary variable (`1` = readmitted within 30 days, `0` = otherwise)

## Key Features Used
- Patient demographics (age, gender, race)
- Hospitalization details (length of stay, number of prior visits)
- Diagnosis codes
- Diabetes medication usage
- Insulin status
- Discharge disposition and admission type

## Data Preprocessing
- Dropped irrelevant and high-missing columns
- Encoded categorical variables using Label Encoding
- Balanced the training data using SMOTE

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest

## Results
The Random Forest classifier showed the most promising results, with improved recall for identifying patients at risk of readmission. Evaluation metrics include precision, recall, F1-score, and confusion matrix.

## Tools Used
- Python (pandas, scikit-learn, imbalanced-learn, seaborn, matplotlib)
- Jupyter Notebook
- GitHub for version control and documentation

## Next Steps
- Fine-tune the model using hyperparameter optimization
- Test on other patient populations or chronic conditions
- Integrate model predictions into a simple dashboard for nursing teams

## Author
Karen Chang  
Licensed RN | MSIT Candidate | UC Berkeley ML/AI Certificate Program
