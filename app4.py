import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load trained model, encoder, and scaler
model = joblib.load("rf_model.pkl")
encoder = joblib.load("rf_encoder.pkl")
scaler = joblib.load("rf_scaler.pkl")

st.title("Car Price Prediction App")
st.write("Enter car details to predict the price:")

# User inputs
kilometer_driven = st.number_input("Kilometer Driven", min_value=0, step=1000)
body_type = st.text_input("Body Type")
manufacturer = st.text_input("Manufacturer")
car_model = st.text_input("Car Model")
manufacturing_year = st.number_input("Manufacturing Year", min_value=1980, max_value=2025, step=1)
city = st.text_input("City")
fuel_type = st.text_input("Fuel Type")
max_power = st.number_input("Max Power (HP)", min_value=0.0, step=1.0)

# Calculate car age
age_of_car = 2025 - manufacturing_year

if st.button("Predict Price"):
    # Prepare input data
    input_data = pd.DataFrame({
        "Kilometer Driven": [kilometer_driven],
        "Body Type": [body_type],
        "Manufacturer": [manufacturer],
        "Car Model": [car_model],
        "Max Power": [max_power],
        "City": [city],
        "Age of Car": [age_of_car],
        "Fuel Type": [fuel_type]
    })
    
    # Encode categorical features
    input_data_encoded = encoder.transform(input_data)
    
    # Scale numeric features
    num_cols = ["Kilometer Driven", "Max Power", "Age of Car"]
    input_data_encoded[num_cols] = scaler.transform(input_data_encoded[num_cols])
    
    # Predict price
    predicted_price = model.predict(input_data_encoded)[0]
    
    st.success(f"Estimated Car Price: ₹{predicted_price:,.2f}")

    st.write("---")
st.write("Developed with ❤️ using Streamlit")
