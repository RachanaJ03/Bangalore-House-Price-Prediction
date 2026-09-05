import json
import joblib
import pandas as pd
import streamlit as st

model = joblib.load("models/house_price_model.pkl")
with open("models/metadata.json") as f:
    meta = json.load(f)

st.set_page_config(page_title="Bangalore House Price Predictor", page_icon="🏠")
st.title("🏠 Bangalore House Price Prediction")
st.write("Enter property details to estimate the price.")

with st.form("prediction"):
    c1, c2 = st.columns(2)
    with c1:
        area = st.number_input("Area (sq ft)", 300, 10000, 1200, 50)
        bhk = st.number_input("BHK", 1, 10, 2, 1)
        bath = st.number_input("Bathrooms", 1, 10, 2, 1)
        balcony = st.number_input("Balconies", 0, 5, 1, 1)
        parking = st.number_input("Parking spaces", 0, 5, 1, 1)
    with c2:
        location = st.selectbox("Location", meta["locations"])
        furnishing = st.selectbox("Furnishing", meta["furnishing"])
        property_type = st.selectbox("Property type", meta["property_types"])
        age = st.number_input("Property age (years)", 0, 100, 5, 1)
    submit = st.form_submit_button("Predict Price")

if submit:
    row = pd.DataFrame([{
        "area": area, "location": location, "bhk": bhk, "bath": bath,
        "balcony": balcony, "parking": parking, "furnishing": furnishing,
        "property_type": property_type, "age": age,
        "area_per_bhk": area / bhk if bhk else 0,
        "bath_per_bhk": bath / bhk if bhk else 0,
        "age_squared": age ** 2
    }])
    prediction = float(model.predict(row)[0])
    st.success(f"Estimated Price: ₹{prediction:,.0f}")
    st.info(f"Approx. ₹{prediction/100000:.2f} lakh / ₹{prediction/10000000:.2f} crore")
    st.caption(f"Model: {meta['best_model']}")
