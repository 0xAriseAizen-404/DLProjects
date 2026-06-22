import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st

import torch
import torch.nn as nn
class ChurnANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)


# Loading the Torch Model
torch_model = ChurnANN(input_dim=11)
torch_model.load_state_dict(torch.load('best_model_reg.pth'))
torch_model.eval()

# Load the proprocessor
with open('preprocessor_reg.pkl', 'rb') as f:
    load_preprocessor = pickle.load(f)

# Load the Salary Scaler
with open('salary_scaler.pkl', 'rb') as f:
    load_salary_scaler = pickle.load(f)

# Streamlit App
st.title('Customer Salary Prediction')
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
exited = st.selectbox("Exited", ["Yes", "No"])


if st.button("Predict Churn"):
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
    exited = 1 if exited == "Yes" else 0
    new_customer = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "Exited": [exited]
    })
    st.write(new_customer)

    new_customer = load_preprocessor.transform(new_customer)
    
    new_customer = torch.tensor(new_customer, dtype=torch.float32)
    with torch.no_grad():
        pred = torch_model(new_customer)
    
    pred_salary = load_salary_scaler.inverse_transform(pred.cpu().numpy())[0][0]    
    st.subheader("Prediction")
    st.success(f"Estimated Salary: ₹{pred_salary:,.2f}")