import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model, encoder, and scaler
xgb_model = joblib.load("optimized_xgb_model.pkl")
encoder = joblib.load("optimized_encoder.pkl")
scaler = joblib.load("optimized_scaler.pkl")


st.title("🚗 Car Price Prediction App 🚗")
st.write("Enter car details to predict the price:")

# Assuming your dataset is already loaded in df (you can ignore this if you have it already)
df = pd.read_csv("C:/Users/Jeba Jini/Documents/Pro 3 Car Dekho/Data/processed4_data.csv")  # Replace with your dataset

# Get the unique manufacturers, body types, fuel types, transmission types, and cities from the dataset
unique_manufacturers = df['Manufacturer'].unique()
unique_body_types = df['Body Type'].unique()
unique_fuel_types = df['Fuel Type'].unique()
unique_transmissions = df['Transmission'].unique()
unique_cities = df['City'].unique()

# Streamlit Form for User Input
manufacturer = st.selectbox("Manufacturer", unique_manufacturers)
filtered_car_models = df[df['Manufacturer'] == manufacturer]['Car Model'].unique()
car_model_options = ['Select a Car Model'] + list(filtered_car_models)
car_model = st.selectbox("Car Model", car_model_options)

body_type = st.selectbox("Body Type", unique_body_types)
fuel_type = st.selectbox("Fuel Type", unique_fuel_types)
transmission = st.selectbox("Transmission", unique_transmissions)
city = st.selectbox("City", unique_cities)

registration_year = st.number_input("Registration Year", min_value=1980, max_value=2025, step=1)
engine = st.number_input("Engine (CC)", min_value=500, max_value=7000, step=50)
mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=50.0, step=0.1)
torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, step=5)
max_power = st.number_input("Max Power (bhp)", min_value=20, max_value=800, step=5)
km_driven = st.number_input("Kilometer Driven", min_value=1000, max_value=500000, step=500)

# When the "Predict" button is clicked, make the prediction
if st.button("Predict Price"):
    # Prepare input data for prediction
    input_data = {
        'Kilometer Driven': km_driven,
        'Body Type': body_type,
        'Manufacturer': manufacturer,
        'Car Model': car_model,
        'Max Power': max_power,
        'City': city,
        'Registration Year': registration_year,
        'Fuel Type': fuel_type,
        'Engine': engine,
        'Mileage': mileage,
        'Torque': torque,
        'Transmission': transmission
    }

    # Convert input data into a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply the encoder to categorical features
    input_encoded = encoder.transform(input_df)

    # Apply the scaler to numerical features
    num_cols = ['Kilometer Driven', 'Max Power', 'Registration Year', 'Engine', 'Mileage', 'Torque']
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    # Make the prediction using the trained model
    predicted_price = xgb_model.predict(input_encoded)[0]

    # Display the predicted price
    st.write(f"Predicted Listed Price: ₹{predicted_price:,.2f}")

    st.write("---")
st.write("Developed with ❤️ using Streamlit")

