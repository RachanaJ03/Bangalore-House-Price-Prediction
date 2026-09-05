import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/raw/house_prices_bangalore.csv").drop_duplicates()
for c in ["area","bhk","bath","balcony","parking","age","price"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()
df["area_per_bhk"] = df["area"] / df["bhk"].replace(0, np.nan)
df["bath_per_bhk"] = df["bath"] / df["bhk"].replace(0, np.nan)
df["age_squared"] = df["age"] ** 2
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X, y = df.drop(columns="price"), df["price"]
cat = ["location","furnishing","property_type"]
num = [c for c in X.columns if c not in cat]
pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=10),
    "Lasso Regression": Lasso(alpha=100000, max_iter=20000),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=400, max_depth=15, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=.05, max_depth=3, random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
rows, pipes = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", pre), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({"Model": name, "R2": r2_score(y_test,pred),
                 "MAE": mean_absolute_error(y_test,pred),
                 "RMSE": np.sqrt(mean_squared_error(y_test,pred))})
    pipes[name] = pipe

scores = pd.DataFrame(rows).sort_values("R2", ascending=False)
scores.to_csv("models/model_comparison.csv", index=False)
best = scores.iloc[0]["Model"]
joblib.dump(pipes[best], "models/house_price_model.pkl")
with open("models/metadata.json","w") as f:
    json.dump({"best_model":best,
               "locations":sorted(df.location.unique().tolist()),
               "furnishing":sorted(df.furnishing.unique().tolist()),
               "property_types":sorted(df.property_type.unique().tolist()),
               "features":X.columns.tolist()}, f, indent=2)
print(scores.to_string(index=False))
