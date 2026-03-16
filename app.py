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

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 10, 120, 65)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 9000.0, 500.0)
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

input_dict = {col: 0 for col in feature_columns}

input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges
input_dict['SeniorCitizen'] = senior_citizen

yes_no = {'Yes': 1, 'No': 0}
input_dict['Partner'] = yes_no[partner]
input_dict['Dependents'] = yes_no[dependents]
input_dict['PhoneService_Yes'] = 1 if phone_service == 'Yes' else 0
input_dict['PhoneService_No'] = 1 if phone_service == 'No' else 0
input_dict['PaperlessBilling_Yes'] = 1 if paperless_billing == 'Yes' else 0
input_dict['PaperlessBilling_No'] = 1 if paperless_billing == 'No' else 0
input_dict['OnlineSecurity_Yes'] = 1 if online_security == 'Yes' else 0
input_dict['OnlineSecurity_No'] = 1 if online_security == 'No' else 0
input_dict['TechSupport_Yes'] = 1 if tech_support == 'Yes' else 0
input_dict['TechSupport_No'] = 1 if tech_support == 'No' else 0

for service in ['DSL', 'Fiber optic', 'No']:
    col = f'InternetService_{service}'
    if col in input_dict:
        input_dict[col] = 1 if internet_service == service else 0

for c in ['Month-to-month', 'One year', 'Two year']:
    col = f'Contract_{c}'
    if col in input_dict:
        input_dict[col] = 1 if contract == c else 0

for p in ['Electronic check', 'Mailed check',
          'Bank transfer (automatic)', 'Credit card (automatic)']:
    col = f'PaymentMethod_{p}'
    if col in input_dict:
        input_dict[col] = 1 if payment_method == p else 0

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)
prob = model.predict_proba(input_scaled)[0][1]
prediction = model.predict(input_scaled)[0]

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
