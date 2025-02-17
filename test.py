import streamlit as st
import pandas as pd
import pickle

# Load the trained model and encoders
with open("C:/Users/Jeba Jini/Documents/Pro 3 Car Dekho/Data/model1.pkl", "rb") as f:
    saved_data = pickle.load(f)

# Extract model and label encoders
model = saved_data["model"]
label_encoders = saved_data["encoders"]

# Load the dataset
final_data = pd.read_csv("C:/Users/Jeba Jini/Documents/Pro 3 Car Dekho/Data/Cleaned1_data.csv")

# Sidebar filters
st.sidebar.header("Filter Options")

# Adding Body Type as a filter
body_types = final_data["Body Type"].unique()
selected_body_type = st.sidebar.selectbox("Select Body Type", body_types)

car_models = final_data["Car Model"].unique()
selected_model = st.sidebar.selectbox("Select Car Model", car_models)

min_price, max_price = int(final_data["Listed Price"].min()), int(final_data["Listed Price"].max())
price_range = st.sidebar.slider("Select Price Range", min_price, max_price, (min_price, max_price))

min_year, max_year = int(final_data["Year of Manufacture"].min()), int(final_data["Year of Manufacture"].max())
year = st.sidebar.slider("Select Manufacturing Year", min_year, max_year, (min_year, max_year))

min_engine, max_engine = int(final_data["Engine Displacement"].min()), int(final_data["Engine Displacement"].max())
engine_range = st.sidebar.slider("Select Engine Displacement Range", min_engine, max_engine, (min_engine, max_engine))

min_reg_year, max_reg_year = int(final_data["Registration Year"].min()), int(final_data["Registration Year"].max())
registration_year = st.sidebar.slider("Select Registration Year", min_reg_year, max_reg_year, (min_reg_year, max_reg_year))

# Filter dataset
filtered_data = final_data[
    (final_data["Car Model"] == selected_model) &
    (final_data["Body Type"] == selected_body_type) &  # Added filter for Body Type
    (final_data["Listed Price"].between(price_range[0], price_range[1])) &
    (final_data["Year of Manufacture"].between(year[0], year[1])) &
    (final_data["Engine Displacement"].between(engine_range[0], engine_range[1])) &
    (final_data["Registration Year"].between(registration_year[0], registration_year[1]))
]

st.title("Used Car Price Prediction")
st.write("### Filtered Data")
st.dataframe(filtered_data)

# Prediction function
def predict_price(features):
    # Ensure features have all the required columns (same as when the model was trained)
    for col, le in label_encoders.items():
        if col in features.columns:
            features[col] = le.transform(features[col])  # Transform the feature using the saved encoder
    return model.predict(features)[0]  # Return a single prediction

st.write("### Predict Future Price")

# Get user input
if not filtered_data.empty:
    sample_car = filtered_data.iloc[0]  # Taking the first car from filtered results
    features = sample_car.drop(["Listed Price"])  # Dropping the target column for prediction
    features = features.values.reshape(1, -1)  # Reshaping to match model's expected input

    # Convert to DataFrame to ensure columns match model training
    features_df = pd.DataFrame(features, columns=sample_car.drop(["Listed Price"]).index)

    predicted_price = predict_price(features_df)
    st.write(f"#### Predicted Price: ₹{predicted_price:,.2f}")
else:
    st.write("No matching cars found. Try adjusting the filters.")

st.write("---")
st.write("Developed with ❤️ using Streamlit")
