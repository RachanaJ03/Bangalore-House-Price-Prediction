# Bangalore House Price Prediction System

Machine-learning regression project for estimating Bangalore residential property prices.

## Dataset
Uploaded Bangalore dataset: 1,000 rows and 10 columns.
Target: `price`

Input features:
`area`, `location`, `bhk`, `bath`, `balcony`, `parking`, `furnishing`, `property_type`, `age`

Engineered features:
`area_per_bhk`, `bath_per_bhk`, `age_squared`

## Models
Linear Regression, Ridge, Lasso, Decision Tree, Random Forest and Gradient Boosting.

## Current result
Best model on a fixed 80/20 test split: **Ridge Regression**
R² = **0.6304**
MAE = **₹2,678,020**
RMSE = **₹3,317,929**

These metrics describe this uploaded dataset/test split and are not a guarantee of real-world market accuracy.

## Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Structure
- `data/raw/` original dataset
- `data/processed/` cleaned dataset
- `models/` trained model + comparison + metadata
- `src/` training and prediction code
- `app.py` Streamlit UI
- `README.md`
