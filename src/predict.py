import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")

def predict_price(area, location, bhk, bath, balcony, parking, furnishing, property_type, age):
    row = pd.DataFrame([{
        "area": area, "location": location, "bhk": bhk, "bath": bath,
        "balcony": balcony, "parking": parking, "furnishing": furnishing,
        "property_type": property_type, "age": age,
        "area_per_bhk": area / bhk if bhk else 0,
        "bath_per_bhk": bath / bhk if bhk else 0,
        "age_squared": age ** 2
    }])
    return float(model.predict(row)[0])
