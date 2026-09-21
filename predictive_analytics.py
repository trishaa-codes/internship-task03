"""
Predictive Analytics Using Historical Data
Forecast monthly sales using lag features and Random Forest Regression.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "historical_sales_data.csv"

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# Clean missing sales values
df["Sales"] = df["Sales"].interpolate().bfill().ffill()

# Feature engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["TimeIndex"] = np.arange(len(df))

for lag in [1, 3, 6, 12]:
    df[f"Lag_{lag}"] = df["Sales"].shift(lag)

df["RollingMean_3"] = df["Sales"].rolling(3).mean()
df["RollingMean_6"] = df["Sales"].rolling(6).mean()

df = df.dropna().reset_index(drop=True)

features = [
    "Year", "Month", "TimeIndex", "Promotion",
    "Lag_1", "Lag_3", "Lag_6", "Lag_12",
    "RollingMean_3", "RollingMean_6"
]

# Time-based split
split = int(len(df) * 0.80)
train = df.iloc[:split]
test = df.iloc[split:]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

model.fit(train[features], train["Sales"])
predictions = model.predict(test[features])

mae = mean_absolute_error(test["Sales"], predictions)
rmse = np.sqrt(mean_squared_error(test["Sales"], predictions))
r2 = r2_score(test["Sales"], predictions)
mape = np.mean(np.abs((test["Sales"] - predictions) / test["Sales"])) * 100

print("Model Evaluation")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
