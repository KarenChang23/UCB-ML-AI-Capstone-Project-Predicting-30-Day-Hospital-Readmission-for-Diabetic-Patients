
## 🧪 Inference Guide

This document provides step-by-step instructions for using the trained model to predict 30-day hospital readmission for diabetic patients.

---

### 📦 Load the Trained Model

Ensure the model file (e.g., `readmitted_binary_XGB.joblib`) is located in the `/models/` directory.

```python
import joblib

# Load the trained model
model = joblib.load("models/readmitted_binary_XGB.joblib")
```

---

### 🧾 Input Format

To ensure accurate predictions, the input data must be **preprocessed** in the same way as the training set:

- **Categorical variables** (e.g., `age`, `race`, `insulin`, `A1Cresult`) must be **label encoded**.
- Columns must be in the **same order and format** as used during training.
- All required features should be present, even if values are `0` or `None`.

📌 Required features include:

- `age`, `race`, `gender`
- `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
- `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `number_outpatient`, etc.
- `insulin`, `A1Cresult`, `change`, `diabetesMed`
- Pre-engineered target column: `readmitted_binary` (only for testing)

---

### 🧠 Making a Prediction

```python
import pandas as pd

# Example single-patient input (replace with actual values)
X_new = pd.DataFrame([{
    'age': 5,
    'race': 2,
    'gender': 1,
    'admission_type_id': 1,
    'discharge_disposition_id': 1,
    'time_in_hospital': 3,
    'insulin': 1,
    'A1Cresult': 1,
    # Add all remaining features...
}])

# Make prediction
y_pred = model.predict(X_new)
y_prob = model.predict_proba(X_new)

print("Predicted:", y_pred)
print("Probability:", y_prob)
```

---

### 📈 Output Explanation

- `y_pred`:  
  - `1` → Patient is likely to be readmitted within 30 days  
  - `0` → Not likely to be readmitted

- `y_prob`:  
  - Probability scores for both classes (e.g., `[0.30, 0.70]` means 70% chance of readmission)

---

### 🚀 Optional Deployment (e.g., Streamlit)

You can deploy the model as an interactive prediction app using [Streamlit](https://streamlit.io/).

```bash
pip install streamlit
streamlit run predict_app.py
```

Example usage script can include:

```python
st.title("Hospital Readmission Predictor")
age = st.selectbox("Age", options=[1, 2, 3, ...])
# Collect other features...
if st.button("Predict"):
    prediction = model.predict(pd.DataFrame([...]))
    st.write("Prediction:", prediction)
```

---

### 📁 File Structure

```
Capstone-Readmission-Prediction/
├── models/
│   └── readmitted_binary_XGB.joblib
├── notebooks/
│   └── Capstone_Readmission_Enhanced_Notebook.ipynb
├── INFERENCE.md
└── ...
```
