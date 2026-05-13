import streamlit as st
import pandas as pd
import joblib

# 1. Load the Saved Pipeline and Feature Columns
model = joblib.load('bank_model_pipeline.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 2. Web Page Configuration
st.set_page_config(page_title="Bank Subscription Prediction System", layout="centered")
st.title("🏦 Bank Deposit Subscription Prediction")
st.write("Enter customer details to predict whether they will subscribe to a term deposit.")

# 3. User Input Fields (Sidebar)
st.sidebar.header("Customer Information")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.sidebar.number_input("Last Contact Duration (Seconds)", min_value=0, value=200)
campaign = st.sidebar.number_input("Number of Contacts during Campaign", min_value=1, value=1)
pdays = st.sidebar.number_input("Days Since Last Contact (999=Never contacted)", value=999)
previous = st.sidebar.number_input("Previous Contacts (Before Campaign)", min_value=0, value=0)

# 4. Prediction Button and Logic
if st.button("Predict"):
    # Create input data frame from user inputs
    input_data = pd.DataFrame([[age, duration, campaign, pdays, previous]], 
                              columns=['age', 'duration', 'campaign', 'pdays', 'previous'])
    
    # Fill missing columns (jobs, months, etc.) with 0
    for col in feature_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    
    # Reorder columns to match the model training order
    input_data = input_data[feature_columns]
    
    # Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    # Display Results
    st.divider()
    if prediction[0] == 1:
        st.success(f"🎉 Result: Customer WILL SUBSCRIBE! (Probability: {probability*100:.2f}%)")
    else:
        st.error(f"❌ Result: Customer WILL NOT SUBSCRIBE. (Probability: {(1-probability)*100:.2f}%)")
