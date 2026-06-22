import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
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

# Loading the Keras Model
tf_model = tf.keras.models.load_model('ANN_tf.keras')

# Loading the Torch Model
torch_model = ChurnANN(input_dim=11)
torch_model.load_state_dict(torch.load('best_model.pth'))
torch_model.eval()

# Load the proprocessor
with open('preprocessor.pkl', 'rb') as f:
    load_preprocessor = pickle.load(f)

# Streamlit App
st.title('Customer Churn Prediction')
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0,value=100000.0)


if st.button("Predict Churn"):
    has_cr_card = 1 if has_cr_card == "Yes" else 0
    is_active_member = 1 if is_active_member == "Yes" else 0
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
        "EstimatedSalary": [estimated_salary]
    })
    st.write(new_customer)

    new_customer = load_preprocessor.transform(new_customer)

    # probability = tf_model.predict(new_customer)[0][0]
    
    new_customer = torch.tensor(new_customer, dtype=torch.float32)
    with torch.no_grad():
        logits = torch_model(new_customer)
        probability = torch.sigmoid(logits).item()
        
    st.subheader("Prediction")
    st.write(f"Churn Probability: {probability:.4f}%")
    if probability >= 0.5:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is gonna Stay")