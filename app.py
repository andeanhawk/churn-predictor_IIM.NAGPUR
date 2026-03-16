import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
with open('feature_columns.json') as f:
    feature_columns = json.load(f)

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their profile.")

st.sidebar.header("Enter Customer Details")

senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10, 120, 65)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])

# Build input with all 45 columns set to 0
input_dict = {col: 0 for col in feature_columns}

# Numerical
input_dict['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

# Gender
input_dict['gender_Female'] = 1 if gender == "Female" else 0
input_dict['gender_Male'] = 1 if gender == "Male" else 0

# Partner
input_dict['Partner_Yes'] = 1 if partner == "Yes" else 0
input_dict['Partner_No'] = 1 if partner == "No" else 0

# Dependents
input_dict['Dependents_Yes'] = 1 if dependents == "Yes" else 0
input_dict['Dependents_No'] = 1 if dependents == "No" else 0

# Phone Service
input_dict['PhoneService_Yes'] = 1 if phone_service == "Yes" else 0
input_dict['PhoneService_No'] = 1 if phone_service == "No" else 0

# Multiple Lines
input_dict['MultipleLines_Yes'] = 1 if multiple_lines == "Yes" else 0
input_dict['MultipleLines_No'] = 1 if multiple_lines == "No" else 0
input_dict['MultipleLines_No phone service'] = 1 if multiple_lines == "No phone service" else 0

# Internet Service
input_dict['InternetService_DSL'] = 1 if internet_service == "DSL" else 0
input_dict['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
input_dict['InternetService_No'] = 1 if internet_service == "No" else 0

# Online Security
input_dict['OnlineSecurity_Yes'] = 1 if online_security == "Yes" else 0
input_dict['OnlineSecurity_No'] = 1 if online_security == "No" else 0
input_dict['OnlineSecurity_No internet service'] = 1 if online_security == "No internet service" else 0

# Online Backup
input_dict['OnlineBackup_Yes'] = 1 if online_backup == "Yes" else 0
input_dict['OnlineBackup_No'] = 1 if online_backup == "No" else 0
input_dict['OnlineBackup_No internet service'] = 1 if online_backup == "No internet service" else 0

# Device Protection
input_dict['DeviceProtection_Yes'] = 1 if device_protection == "Yes" else 0
input_dict['DeviceProtection_No'] = 1 if device_protection == "No" else 0
input_dict['DeviceProtection_No internet service'] = 1 if device_protection == "No internet service" else 0

# Tech Support
input_dict['TechSupport_Yes'] = 1 if tech_support == "Yes" else 0
input_dict['TechSupport_No'] = 1 if tech_support == "No" else 0
input_dict['TechSupport_No internet service'] = 1 if tech_support == "No internet service" else 0

# Streaming TV
input_dict['StreamingTV_Yes'] = 1 if streaming_tv == "Yes" else 0
input_dict['StreamingTV_No'] = 1 if streaming_tv == "No" else 0
input_dict['StreamingTV_No internet service'] = 1 if streaming_tv == "No internet service" else 0

# Streaming Movies
input_dict['StreamingMovies_Yes'] = 1 if streaming_movies == "Yes" else 0
input_dict['StreamingMovies_No'] = 1 if streaming_movies == "No" else 0
input_dict['StreamingMovies_No internet service'] = 1 if streaming_movies == "No internet service" else 0

# Contract
input_dict['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
input_dict['Contract_One year'] = 1 if contract == "One year" else 0
input_dict['Contract_Two year'] = 1 if contract == "Two year" else 0

# Paperless Billing
input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == "Yes" else 0
input_dict['PaperlessBilling_No'] = 1 if paperless_billing == "No" else 0

# Payment Method
input_dict['PaymentMethod_Electronic check'] = 1 if payment_method == "Electronic check" else 0
input_dict['PaymentMethod_Mailed check'] = 1 if payment_method == "Mailed check" else 0
input_dict['PaymentMethod_Bank transfer (automatic)'] = 1 if payment_method == "Bank transfer (automatic)" else 0
input_dict['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == "Credit card (automatic)" else 0

# Predict
input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

# Display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{round(prob * 100, 1)}%")

with col2:
    risk = "High Risk" if prob > 0.7 else "Medium Risk" if prob > 0.4 else "Low Risk"
    st.metric("Risk Level", risk)

with col3:
    st.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")

st.markdown("### Churn Risk Meter")
st.progress(float(prob))

if prob > 0.7:
    st.error("High churn risk! Consider offering a discount or contract upgrade.")
elif prob > 0.4:
    st.warning("Medium risk. Monitor this customer closely.")
else:
    st.success("Low churn risk. Customer looks healthy!")
