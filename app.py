import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open("model.pkl", "rb"))

st.title("💳 Credit Scoring Predictor")

# Input fields
log_income = st.number_input("Log Income", min_value=0.0)
log_loan_amount = st.number_input("Log Loan Amount", min_value=0.0)
credit_history = st.selectbox("Credit History", [0, 1])
term_binary = st.selectbox("Loan Term (0 = short, 1 = long)", [0, 1])

if st.button("Predict"):
    input_data = pd.DataFrame([[log_income, log_loan_amount, credit_history, term_binary]],
                              columns=['log_income', 'log_loan_amount', 'credit_history', 'term_binary'])
    prediction = model.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == 0 else "❌ Loan Rejected"
    st.success(result)
