


base_path = "."

import pandas as pd

april = pd.read_parquet(f"{base_path}/hourly_demand_04.parquet")
may = pd.read_parquet(f"{base_path}/hourly_demand_05.parquet")
june = pd.read_parquet(f"{base_path}/hourly_demand_06.parquet")

trip_df = pd.concat(
    [april, may, june],
    ignore_index=True
)

print(trip_df.shape)
print(trip_df.columns.tolist())
import requests
import pandas as pd

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "start_date": "2021-04-01",
    "end_date": "2021-06-30",
    "hourly": [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "precipitation",
        "snowfall",
        "windspeed_10m",
        "windgusts_10m",
        "surface_pressure",
        "cloudcover",
        "visibility"
    ],
    "timezone": "America/New_York"
}

resp = requests.get(url, params=params)
resp.raise_for_status()
data = resp.json()["hourly"]

hourly_weather_df = pd.DataFrame(data)
hourly_weather_df["time"] = pd.to_datetime(hourly_weather_df["time"])

print(hourly_weather_df.shape)
print(hourly_weather_df.head())
hourly_weather_df = hourly_weather_df.drop(columns=["visibility"])
hourly_weather_df = hourly_weather_df.rename(columns={
    "time": "datetime",
    "temperature_2m": "temp",
    "apparent_temperature": "feelslike",
    "relative_humidity_2m": "humidity",
    "precipitation": "precip",
    "snowfall": "snow",
    "windspeed_10m": "windspeed",
    "windgusts_10m": "windgust",
    "surface_pressure": "sealevelpressure",
    "cloudcover": "cloudcover"
})
hourly_weather_df["precipprob"] = (hourly_weather_df["precip"] > 0).astype(int) * 100
hourly_weather_df["day"] = hourly_weather_df["datetime"].dt.day
hourly_weather_df["month"] = hourly_weather_df["datetime"].dt.month
hourly_weather_df["dayofweek"] = hourly_weather_df["datetime"].dt.dayofweek
hourly_weather_df["weekofyear"] = hourly_weather_df["datetime"].dt.isocalendar().week
hourly_weather_df["is_weekend"] = hourly_weather_df["dayofweek"].isin([5, 6]).astype(int)
hourly_weather_df["quarter"] = hourly_weather_df["datetime"].dt.quarter
hourly_weather_df["dayofyear"] = hourly_weather_df["datetime"].dt.dayofyear
weather_df_daily = pd.read_parquet(
    f"{base_path}/weather_cleaned.parquet"
)
# ==========================================
# Load cleaned DAILY weather data (kept for comparison)
# ==========================================

weather_df_daily = pd.read_parquet(
    f"{base_path}/weather_cleaned.parquet"
)

print("Daily weather shape:", weather_df_daily.shape)
print("Daily weather columns:")
print(weather_df_daily.columns.tolist())
zone_df = pd.read_csv(
    f"{base_path}/taxi_zone_lookup.csv"
)

print("Zone shape:", zone_df.shape)
print(zone_df.columns.tolist())
# Align both timestamps to the hour before merging
trip_df["pickup_hour"] = pd.to_datetime(trip_df["pickup_hour"]).dt.floor("h")
hourly_weather_df["datetime"] = pd.to_datetime(hourly_weather_df["datetime"]).dt.floor("h")

ride_weather_df = trip_df.merge(
    hourly_weather_df,
    left_on="pickup_hour",
    right_on="datetime",
    how="left"
)

print("ride_weather_df shape:", ride_weather_df.shape)
print(ride_weather_df.head())
print("Missing weather values:", ride_weather_df["temp"].isna().sum())

sample_day = ride_weather_df[ride_weather_df["pickup_hour"].dt.date.astype(str) == "2021-04-01"]
print("Unique temp values on 2021-04-01:", sample_day["temp"].nunique())
combined_df = ride_weather_df.merge(
    zone_df,
    left_on="PULocationID",
    right_on="LocationID",
    how="left"
)

print("combined_df shape:", combined_df.shape)
print(combined_df.head())
# Check unmatched pickup location IDs

missing_locations = combined_df.loc[
    combined_df["Zone"].isna(),
    "PULocationID"
].unique()

print("Number of unmatched LocationIDs:", len(missing_locations))
print("Unmatched LocationIDs:")
print(sorted(missing_locations))
# ==========================================
# Create final modeling dataframe
# ==========================================

model_df = combined_df.copy()

# Create pickup_date directly from pickup_hour
model_df["pickup_date"] = pd.to_datetime(
    model_df["pickup_hour"]
).dt.normalize()

# Sort chronologically
model_df = model_df.sort_values(
    ["pickup_date", "pickup_hour", "PULocationID"]
).reset_index(drop=True)

print("Date range:")
print(
    model_df["pickup_date"].min(),
    "to",
    model_df["pickup_date"].max()
)

print("\nModel dataframe shape:")
print(model_df.shape)

print("\nMissing pickup_date values:")
print(model_df["pickup_date"].isna().sum())
features = [
    "pickup_hour",
    "PULocationID",
    "temp",
    "feelslike",
    "humidity",
    "precip",
    "precipprob",
    "snow",
    "windgust",
    "windspeed",
    "sealevelpressure",
    "cloudcover",
    "day",
    "month",
    "dayofweek",
    "weekofyear",
    "is_weekend",
    "quarter",
    "dayofyear"
]

X = model_df[features]
y = model_df["ride_count"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing values in X:", X.isna().sum().sum())
print("Missing values in y:", y.isna().sum())
# ==========================================
# STEP 3: Time-based train/test split
# ==========================================

train_mask = model_df["pickup_date"] < "2021-06-01"
test_mask = model_df["pickup_date"] >= "2021-06-01"

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining period:")
print(
    model_df.loc[train_mask, "pickup_date"].min(),
    "to",
    model_df.loc[train_mask, "pickup_date"].max()
)

print("\nTesting period:")
print(
    model_df.loc[test_mask, "pickup_date"].min(),
    "to",
    model_df.loc[test_mask, "pickup_date"].max()
)
# ==========================================
# Convert pickup_hour to numeric hour
# ==========================================

X_train = X_train.copy()
X_test = X_test.copy()

X_train["pickup_hour"] = X_train["pickup_hour"].dt.hour
X_test["pickup_hour"] = X_test["pickup_hour"].dt.hour

print("pickup_hour dtype:", X_train["pickup_hour"].dtype)

print("Missing values in X_train:", X_train.isna().sum().sum())
print("Missing values in X_test:", X_test.isna().sum().sum())
# ==========================================
# STEP 4: Train Random Forest
# ==========================================

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest model...")

rf_model.fit(X_train, y_train)

print("Random Forest training completed!")
# ==========================================
# STEP 5: Make predictions
# ==========================================

y_pred = rf_model.predict(X_test)

print("Prediction completed!")
print("Number of predictions:", len(y_pred))

print("\nFirst 10 actual values:")
print(y_test.values[:10])

print("\nFirst 10 predicted values:")
print(y_pred[:10])
print(model_df[["pickup_hour", "PULocationID", "ride_count"]].head(20))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.3f}")

#Weather feature importance
importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)

weather_features = [
    "temp", "feelslike", "humidity", "precip", "precipprob",
    "snow", "windgust", "windspeed", "sealevelpressure", "cloudcover"
]

weather_importance = importances[weather_features].sort_values(ascending=False)
print(weather_importance)
print(f"\nTotal weather importance: {weather_importance.sum():.4f} ({weather_importance.sum()*100:.2f}%)")
non_weather_features = [
    "pickup_hour", "PULocationID",
    "day", "month", "dayofweek", "weekofyear",
    "is_weekend", "quarter", "dayofyear"
]

X_train_nw = X_train[non_weather_features]
X_test_nw = X_test[non_weather_features]

rf_nw = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_nw.fit(X_train_nw, y_train)
y_pred_nw = rf_nw.predict(X_test_nw)

rmse_nw = np.sqrt(mean_squared_error(y_test, y_pred_nw))
r2_nw = r2_score(y_test, y_pred_nw)

print(f"No-weather RMSE: {rmse_nw:.2f}  (with hourly weather: 24.90)")
print(f"No-weather R²:   {r2_nw:.3f}  (with hourly weather: 0.911)")
dupe_check = trip_df.duplicated(subset=["pickup_hour", "PULocationID"]).sum()
print("Duplicate (location, hour) rows:", dupe_check)
#26
# Add lag + rolling + cyclical features to the HOURLY weather model_df

model_df = model_df.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)

# Lag features: same location, N hours ago
for lag in [1, 24, 168]:
    model_df[f"lag_{lag}"] = (
        model_df.groupby("PULocationID")["ride_count"].shift(lag)
    )

# Rolling mean (24h), shifted so it doesn't leak the current hour
model_df["rolling_mean_24"] = (
    model_df.groupby("PULocationID")["ride_count"]
    .transform(lambda s: s.shift(1).rolling(24).mean())
)

# Cyclical encoding for hour-of-day and day-of-week
model_df["hour_of_day"] = model_df["pickup_hour"].dt.hour
model_df["hour_sin"] = np.sin(2 * np.pi * model_df["hour_of_day"] / 24)
model_df["hour_cos"] = np.cos(2 * np.pi * model_df["hour_of_day"] / 24)
model_df["dow_sin"] = np.sin(2 * np.pi * model_df["dayofweek"] / 7)
model_df["dow_cos"] = np.cos(2 * np.pi * model_df["dayofweek"] / 7)

# Drop rows with no lag history yet (first week per location)
print("Rows before dropping NaN history rows:", len(model_df))
model_df = model_df.dropna(subset=["lag_1", "lag_24", "lag_168", "rolling_mean_24"])
print("Rows after:", len(model_df))
#27
features = [
    "hour_of_day", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "PULocationID",
    "temp", "feelslike", "humidity", "precip", "precipprob",
    "snow", "windgust", "windspeed", "sealevelpressure", "cloudcover",
    "day", "month", "weekofyear", "is_weekend", "quarter", "dayofyear",
    "lag_1", "lag_24", "lag_168", "rolling_mean_24"
]

X = model_df[features]
y = model_df["ride_count"]

train_mask = model_df["pickup_date"] < "2021-06-01"
test_mask = model_df["pickup_date"] >= "2021-06-01"

X_train = X[train_mask].copy()
X_test = X[test_mask].copy()
y_train = y[train_mask]
y_test = y[test_mask]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
import joblib
joblib.dump(rf_model, "rf_model.pkl")
print("Saved rf_model.pkl!")
y_pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "r2": float(r2),
    "feature_importances": [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in sorted(
            zip(X_train.columns, rf_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ],
}
import json
with open("metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump(metrics, metrics_file, indent=2)
print("Saved metrics.json")
print(f"Hourly weather + lags — RMSE: {rmse:.2f}, R²: {r2:.3f}")

import holidays
us_holidays = holidays.US(years=[2021])
print(us_holidays)
holiday_dates = set(us_holidays.keys())

model_df["is_holiday"] = model_df["pickup_date"].dt.date.apply(
    lambda x: int(x in holiday_dates)
)

model_df["is_day_before_holiday"] = model_df["pickup_date"].dt.date.apply(
    lambda x: int((x + pd.Timedelta(days=1)) in holiday_dates)
)
model_df[model_df["is_holiday"] == 1][
    ["pickup_date", "is_holiday"]
].drop_duplicates()
model_df[model_df["is_day_before_holiday"] == 1][
    ["pickup_date", "is_day_before_holiday"]
].drop_duplicates()
print(model_df["is_holiday"].value_counts())
print(model_df["is_day_before_holiday"].value_counts())
model_df.loc[
    (model_df["is_holiday"] == 1) |
    (model_df["is_day_before_holiday"] == 1),
    ["pickup_date", "is_holiday", "is_day_before_holiday"]
].drop_duplicates().sort_values("pickup_date")
# ==========================================
# EXPERIMENT: Add Holiday Features
# ==========================================

holiday_features = features + [
    "is_holiday",
    "is_day_before_holiday"
]

X_holiday = model_df[holiday_features]
y_holiday = model_df["ride_count"]

# Same time-based train/test split
train_mask = model_df["pickup_date"] < "2021-06-01"
test_mask = model_df["pickup_date"] >= "2021-06-01"

X_train_holiday = X_holiday[train_mask].copy()
X_test_holiday = X_holiday[test_mask].copy()

y_train_holiday = y_holiday[train_mask]
y_test_holiday = y_holiday[test_mask]

print("X_train:", X_train_holiday.shape)
print("X_test:", X_test_holiday.shape)
print(X_train_holiday.columns.tolist())
rf_holiday_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_holiday_model.fit(
    X_train_holiday,
    y_train_holiday
)

y_pred_holiday = rf_holiday_model.predict(X_test_holiday)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

rmse_holiday = np.sqrt(
    mean_squared_error(y_test_holiday, y_pred_holiday)
)

mae_holiday = mean_absolute_error(
    y_test_holiday, y_pred_holiday
)

r2_holiday = r2_score(
    y_test_holiday, y_pred_holiday
)

print("HOLIDAY MODEL")
print(f"RMSE: {rmse_holiday:.2f}")
print(f"MAE:  {mae_holiday:.2f}")
print(f"R²:   {r2_holiday:.3f}")
holiday_mask_test = X_test_holiday["is_holiday"] == 1

rmse_holiday_only_old = np.sqrt(
    mean_squared_error(y_test[holiday_mask_test], y_pred[holiday_mask_test])
)
rmse_holiday_only_new = np.sqrt(
    mean_squared_error(y_test_holiday[holiday_mask_test], y_pred_holiday[holiday_mask_test])
)

print(f"Holiday-only RMSE — before holiday feature: {rmse_holiday_only_old:.2f}")
print(f"Holiday-only RMSE — after holiday feature:  {rmse_holiday_only_new:.2f}")
print(f"Number of holiday rows in test set: {holiday_mask_test.sum()}")
print(model_df[model_df["is_holiday"]==1]["pickup_date"].unique())
