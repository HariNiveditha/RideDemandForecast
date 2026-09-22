import json
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_model.pkl"
HISTORY_PATH = BASE_DIR / "ride_demand_weather_hourly.parquet"
FEATURE_ORDER = [
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
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
    "weekofyear",
    "is_weekend",
    "quarter",
    "dayofyear",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
]
WEATHER_FIELDS = [
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

model = None
history = None
history_by_key = None


class PredictionRequest(BaseModel):
    PULocationID: int = Field(gt=0)
    datetime: datetime
    temp: float
    feelslike: float
    humidity: float
    precip: float
    precipprob: float
    snow: float
    windgust: float
    windspeed: float
    sealevelpressure: float
    cloudcover: float


def _load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        raise RuntimeError(f"Historical ride data was not found at {HISTORY_PATH.name}.")

    frame = pd.read_parquet(HISTORY_PATH, columns=["pickup_hour", "PULocationID", "ride_count"])
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"]).dt.floor("h")
    frame["PULocationID"] = frame["PULocationID"].astype(int)
    return frame.set_index(["PULocationID", "pickup_hour"]).sort_index()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, history, history_by_key
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if HISTORY_PATH.exists():
        history = _load_history()
        history_by_key = history["ride_count"]
    yield


app = FastAPI(title="RideCast prediction API", version="1.0.0", lifespan=lifespan)


@app.get("/metrics")
def get_metrics() -> dict:
    metrics_path = BASE_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json was not found.")

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _naive_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.floor("h")


def _lookup_lags(location_id: int, timestamp: pd.Timestamp) -> tuple[float, float, float, float]:
    if history_by_key is None:
        raise HTTPException(status_code=503, detail="Historical ride data is not available.")

    location_values = history_by_key.index.get_level_values("PULocationID")
    if location_id not in location_values:
        raise HTTPException(status_code=400, detail="PULocationID does not exist in historical ride data.")

    lag_values = []
    for hours in (1, 24, 168):
        key = (location_id, timestamp - pd.Timedelta(hours=hours))
        try:
            lag_values.append(float(history_by_key.loc[key]))
        except KeyError as error:
            raise HTTPException(status_code=400, detail=f"Historical data is unavailable for lag_{hours} at this location and time.") from error

    rolling_values = []
    for hours in range(1, 25):
        key = (location_id, timestamp - pd.Timedelta(hours=hours))
        try:
            rolling_values.append(float(history_by_key.loc[key]))
        except KeyError as error:
            raise HTTPException(status_code=400, detail="Historical data is insufficient for rolling_mean_24 at this location and time.") from error

    return (*lag_values, float(np.mean(rolling_values)))


def _build_features(request: PredictionRequest) -> pd.DataFrame:
    timestamp = _naive_timestamp(request.datetime)
    lag_1, lag_24, lag_168, rolling_mean_24 = _lookup_lags(request.PULocationID, timestamp)
    iso_week = timestamp.isocalendar().week
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek

    values = {
        "hour_of_day": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos": np.cos(2 * np.pi * day_of_week / 7),
        "PULocationID": request.PULocationID,
        **{field: getattr(request, field) for field in WEATHER_FIELDS},
        "day": timestamp.day,
        "month": timestamp.month,
        "weekofyear": int(iso_week),
        "is_weekend": int(day_of_week in (5, 6)),
        "quarter": timestamp.quarter,
        "dayofyear": timestamp.dayofyear,
        "lag_1": lag_1,
        "lag_24": lag_24,
        "lag_168": lag_168,
        "rolling_mean_24": rolling_mean_24,
    }
    return pd.DataFrame([[values[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)


def _build_model_frame() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        raise RuntimeError(f"Historical ride data was not found at {HISTORY_PATH.name}.")

    frame = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "pickup_hour",
            "PULocationID",
            "ride_count",
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
            "dayofyear",
        ],
    )
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"]).dt.floor("h")
    frame["pickup_date"] = frame["pickup_hour"].dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)

    for lag in (1, 24, 168):
        frame[f"lag_{lag}"] = frame.groupby("PULocationID")["ride_count"].shift(lag)

    frame["rolling_mean_24"] = (
        frame.groupby("PULocationID")["ride_count"]
        .transform(lambda series: series.shift(1).rolling(24).mean())
    )

    frame["hour_of_day"] = frame["pickup_hour"].dt.hour
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour_of_day"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour_of_day"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)

    frame = frame.dropna(subset=["lag_1", "lag_24", "lag_168", "rolling_mean_24"]).copy()
    return frame


@lru_cache(maxsize=1)
def _build_prediction_sample_points() -> list[dict]:
    if model is None:
        raise HTTPException(status_code=503, detail="The serialized model is not available.")

    frame = _build_model_frame()
    test_frame = frame.loc[frame["pickup_date"] >= "2021-06-01"].sort_values("pickup_hour").copy()
    if test_frame.empty:
        raise HTTPException(status_code=404, detail="No test rows were found for the prediction sample.")

    X_test = test_frame.loc[:, FEATURE_ORDER]
    y_test = test_frame["ride_count"].to_numpy(dtype=float)
    predictions = model.predict(X_test)

    sample_size = min(80, len(y_test))
    sample_indices = np.linspace(0, len(y_test) - 1, num=sample_size, dtype=int)
    sample_frame = test_frame.iloc[sample_indices].copy()
    sample_predictions = predictions[sample_indices]

    return [
        {
            "timestamp": row["pickup_hour"].isoformat(),
            "actual": float(row["ride_count"]),
            "predicted": float(prediction),
        }
        for row, prediction in zip(sample_frame.to_dict("records"), sample_predictions)
    ]


@app.get("/predictions-sample")
def get_predictions_sample() -> dict:
    return {"points": _build_prediction_sample_points()}


@app.get("/historical")
def get_historical() -> dict:
    if not HISTORY_PATH.exists():
        raise HTTPException(status_code=404, detail="Historical ride data was not found.")

    frame = pd.read_parquet(HISTORY_PATH, columns=["pickup_hour", "PULocationID", "ride_count"])
    frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"]).dt.floor("h")
    frame["PULocationID"] = frame["PULocationID"].astype(int)

    overall = (
        frame.groupby("pickup_hour", as_index=False)["ride_count"]
        .sum()
        .rename(columns={"ride_count": "total_rides"})
        .sort_values("pickup_hour")
        .reset_index(drop=True)
    )

    demand_by_hour_of_day = (
        frame.assign(hour=frame["pickup_hour"].dt.hour)
        .groupby("hour", as_index=False)["ride_count"]
        .mean()
        .rename(columns={"ride_count": "avg_rides"})
        .sort_values("hour")
        .reset_index(drop=True)
    )

    demand_by_location = (
        frame.groupby("PULocationID", as_index=False)["ride_count"]
        .sum()
        .rename(columns={"ride_count": "total_rides"})
        .sort_values("total_rides", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    peak_overall = overall.loc[overall["total_rides"].idxmax()]
    peak_location = demand_by_location.iloc[0]
    avg_demand = float(overall["total_rides"].mean())

    response = {
        "date_range": {
            "start": overall["pickup_hour"].min().isoformat(),
            "end": overall["pickup_hour"].max().isoformat(),
        },
        "summary": {
            "avg_demand": avg_demand,
            "peak_demand": float(peak_overall["total_rides"]),
            "peak_at": peak_overall["pickup_hour"].isoformat(),
            "peak_location": int(peak_location["PULocationID"]),
        },
        "hourly_series": [
            {"pickup_hour": row["pickup_hour"].isoformat(), "total_rides": float(row["total_rides"])}
            for _, row in overall.iterrows()
        ],
        "demand_by_hour_of_day": [
            {"hour": int(row["hour"]), "avg_rides": float(row["avg_rides"])}
            for _, row in demand_by_hour_of_day.iterrows()
        ],
        "demand_by_location": [
            {"PULocationID": int(row["PULocationID"]), "total_rides": float(row["total_rides"])}
            for _, row in demand_by_location.iterrows()
        ],
    }

    if len(response["hourly_series"]) > 2000:
        response["size_warning"] = "Hourly series exceeds 2000 points and may be large for browsers to render efficiently."

    return response


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    if model is None:
        raise HTTPException(status_code=503, detail="The serialized model is not available. Run the training notebook serialization cell first.")

    features = _build_features(request)
    prediction = float(model.predict(features)[0])
    return {"predicted_demand": prediction}
