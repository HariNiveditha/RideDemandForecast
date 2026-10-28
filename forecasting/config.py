from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET_PATH = BASE_DIR / "ride_demand_weather_hourly.parquet"
TAXI_ZONE_LOOKUP_PATH = BASE_DIR / "taxi_zone_lookup.csv"
METRICS_JSON_PATH = BASE_DIR / "metrics.json"
ARTIFACTS_DIR = BASE_DIR / "forecasting" / "artifacts"
FIXTURES_DIR = BASE_DIR / "fixtures"
PLOTS_DIR = BASE_DIR / "plots"

# Supported forecast horizons in minutes and corresponding hourly offsets
DEFAULT_HORIZONS_MINUTES = [60, 180, 360]  # 1 hour, 3 hours, 6 hours ahead
HORIZON_STEPS = {
    60: 1,
    180: 3,
    360: 6,
}

# Chronological split timestamps
TRAIN_START = "2021-04-01 00:00:00"
VAL_CUTOFF = "2021-05-16 00:00:00"
TEST_CUTOFF = "2021-06-01 00:00:00"
TEST_END = "2021-06-30 23:00:00"

# Target variable metadata
TARGET_COLUMN = "ride_count"
TARGET_DESCRIPTION = "Total yellow taxi pickups completed in the taxi zone during the hourly interval."

# Base weather features
WEATHER_FEATURES = [
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
]

# Temporal features derived at target timestamp
CALENDAR_FEATURES = [
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "day",
    "month",
    "weekofyear",
    "quarter",
    "dayofyear",
]

# Standard feature order for multi-horizon model
MULTI_HORIZON_FEATURE_ORDER = [
    "target_lag_168",
    "target_lag_24",
    "origin_lag_0",
    "origin_lag_1",
    "origin_rolling_mean_24",
    "origin_rolling_std_24",
    "PULocationID",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "dayofweek",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "day",
    "month",
    "weekofyear",
    "quarter",
    "dayofyear",
    "weather_temp",
    "weather_feelslike",
    "weather_humidity",
    "weather_precip",
    "weather_precipprob",
    "weather_snow",
    "weather_windgust",
    "weather_windspeed",
    "weather_sealevelpressure",
    "weather_cloudcover",
]

MODEL_VERSION = "layer1-multihorizon-v1.0"
DEFAULT_INTERVAL_CONFIDENCE = 0.80  # 80% empirical prediction interval
