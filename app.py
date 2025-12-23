

import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("best_rf_model.pkl")

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="💊", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B9CD3;'>💊 Insurance Out-of-Pocket Cost Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load dataset for dynamic dropdowns
df = pd.read_csv("health_formatted.csv")

# --- Layout with columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient & Hospital Info")
    specialty_name = st.selectbox("Select Specialty", sorted(df['specialty'].unique()))
    procedures = sorted(df[df['specialty'] == specialty_name]['procedure_name'].unique())
    procedure_name = st.selectbox("Select Procedure", procedures)
    procedure_type_name = st.selectbox("Procedure Type", sorted(df['procedure_type'].unique()))
    day_care_name = st.selectbox("Day Care?", sorted(df['day_care'].unique()))

with col2:
    st.subheader("Treatment Details")
    insurance_amount = st.slider("Insurance Amount (₹)", min_value=0, max_value=1000000, value=50000, step=1000)
    severity_name = st.selectbox("Severity Level", ["Low", "Medium", "High"])
    hospital_name = st.selectbox("Hospital Type", ["Government", "Private"])
    length_of_stay = st.slider("Length of Stay (days)", min_value=1, max_value=30, value=5)

st.markdown("---")

# --- Encode categorical features ---
le_specialty = LabelEncoder()
df['specialty_encoded'] = le_specialty.fit_transform(df['specialty'])
specialty_encoded = int(le_specialty.transform([specialty_name])[0])

le_procedure = LabelEncoder()
df['procedure_encoded'] = le_procedure.fit_transform(df['procedure_name'])
procedure_encoded = int(le_procedure.transform([procedure_name])[0])

le_type = LabelEncoder()
df['procedure_type_encoded'] = le_type.fit_transform(df['procedure_type'])
procedure_type_encoded = int(le_type.transform([procedure_type_name])[0])

day_care_encoded = 1 if day_care_name == "Yes" else 0
severity_dict = {"Low":0, "Medium":1, "High":2}
hospital_dict = {"Government":0, "Private":1}
severity_encoded = severity_dict[severity_name]
hospital_encoded = hospital_dict[hospital_name]

# --- Predict ---
if st.button("💰 Predict Out-of-Pocket Cost", type="primary"):
    sample_input = np.array([[specialty_encoded, procedure_encoded, insurance_amount,
                              procedure_type_encoded, day_care_encoded, severity_encoded,
                              hospital_encoded, length_of_stay]])
    predicted_cost = model.predict(sample_input)
    
    # Display prediction in a styled box
    st.markdown(f"""
        <div style="
            background-color:#F0F8FF;
            padding:20px;
            border-radius:10px;
            text-align:center;
            font-size:24px;
            font-weight:bold;
            color:#1E3A8A;
        ">
            Predicted Out-of-Pocket Cost: ₹{predicted_cost[0]:,.2f}
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")


