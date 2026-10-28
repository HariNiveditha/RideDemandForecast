import json
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

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


forecasting_service = None


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

    frame = pd.read_parquet(
        HISTORY_PATH,
        columns=[
            "pickup_hour",
            "PULocationID",
            "ride_count",
            *WEATHER_FIELDS,
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
    test_frame = frame.loc[frame["pickup_hour"] >= pd.Timestamp("2021-06-01")].sort_values("pickup_hour")
    if test_frame.empty:
        raise HTTPException(status_code=404, detail="No test rows were found for the prediction sample.")

    sample_size = min(80, len(test_frame))
    candidate_count = min(len(test_frame), sample_size * 20)
    candidate_indices = np.linspace(0, len(test_frame) - 1, num=candidate_count, dtype=int)
    candidates = test_frame.iloc[candidate_indices].reset_index(drop=True)
    lag_hours = np.array([1, 24, 168, *range(1, 25)])
    query_locations = np.repeat(candidates["PULocationID"].to_numpy(dtype=int), len(lag_hours))
    query_times = np.repeat(candidates["pickup_hour"].to_numpy(), len(lag_hours)) - np.tile(
        pd.to_timedelta(lag_hours, unit="h"), candidate_count
    )
    query_index = pd.MultiIndex.from_arrays(
        [query_locations, query_times], names=history_by_key.index.names
    )
    lag_values = history_by_key.reindex(query_index).to_numpy().reshape(candidate_count, -1)
    valid_indices = np.flatnonzero(np.isfinite(lag_values).all(axis=1))

    if not len(valid_indices):
        raise HTTPException(status_code=404, detail="No prediction sample rows have sufficient lag history.")

    selected_positions = np.linspace(0, len(valid_indices) - 1, num=min(sample_size, len(valid_indices)), dtype=int)
    selected_indices = valid_indices[selected_positions]
    sample_frame = candidates.iloc[selected_indices].copy()
    sample_frame["hour_of_day"] = sample_frame["pickup_hour"].dt.hour
    sample_frame["hour_sin"] = np.sin(2 * np.pi * sample_frame["hour_of_day"] / 24)
    sample_frame["hour_cos"] = np.cos(2 * np.pi * sample_frame["hour_of_day"] / 24)
    sample_frame["dow_sin"] = np.sin(2 * np.pi * sample_frame["dayofweek"] / 7)
    sample_frame["dow_cos"] = np.cos(2 * np.pi * sample_frame["dayofweek"] / 7)
    selected_lags = lag_values[selected_indices]
    sample_frame["lag_1"] = selected_lags[:, 0]
    sample_frame["lag_24"] = selected_lags[:, 1]
    sample_frame["lag_168"] = selected_lags[:, 2]
    sample_frame["rolling_mean_24"] = selected_lags[:, 3:].mean(axis=1)
    sample_predictions = model.predict(sample_frame.loc[:, FEATURE_ORDER])

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


@app.get("/locations")
def get_locations() -> dict:
    lookup_path = BASE_DIR / "taxi_zone_lookup.csv"
    if not lookup_path.exists():
        raise HTTPException(status_code=404, detail="taxi_zone_lookup.csv was not found.")
    if history is None:
        raise HTTPException(status_code=503, detail="Historical ride data is not available.")

    location_ids = set(history.index.get_level_values("PULocationID").unique())
    lookup = pd.read_csv(lookup_path)
    lookup = lookup.loc[lookup["LocationID"].isin(location_ids)].sort_values(["Borough", "Zone"])

    return {
        "locations": [
            {
                "PULocationID": int(row.LocationID),
                "Borough": row.Borough,
                "Zone": row.Zone,
            }
            for row in lookup.itertuples(index=False)
        ]
    }


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


class MultiHorizonForecastRequest(BaseModel):
    zone_id: int = Field(gt=0, description="NYC TLC Taxi Zone ID (PULocationID)")
    forecast_origin: datetime = Field(description="ISO timestamp when forecast is generated")
    horizon_minutes: Optional[int] = Field(default=None, description="Horizon in minutes (60, 180, 360). If omitted, all horizons are returned.")
    temp: Optional[float] = None
    humidity: Optional[float] = None
    precip: Optional[float] = None
    windspeed: Optional[float] = None


@app.get("/forecast/horizons")
def get_supported_horizons() -> dict:
    return {
        "supported_horizons": [
            {"minutes": 60, "hours": 1, "label": "1 Hour Ahead"},
            {"minutes": 180, "hours": 3, "label": "3 Hours Ahead"},
            {"minutes": 360, "hours": 6, "label": "6 Hours Ahead"},
        ],
        "default_horizon": 60,
        "model_version": "layer1-multihorizon-v1.0",
        "has_uncertainty_bounds": True,
    }


@app.post("/forecast/multi-horizon")
def get_multi_horizon_forecast(request: MultiHorizonForecastRequest) -> dict:
    global forecasting_service
    if forecasting_service is None:
        try:
            from forecasting.service import ForecastingService
            forecasting_service = ForecastingService.get_default_service()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Forecasting service unavailable: {exc}")

    origin_dt = pd.Timestamp(request.forecast_origin).floor("h")
    if origin_dt.tzinfo is not None:
        origin_dt = origin_dt.tz_convert(None)

    valid_horizons = [60, 180, 360]
    if request.horizon_minutes is not None:
        if request.horizon_minutes not in valid_horizons:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported horizon_minutes: {request.horizon_minutes}. Supported horizons: {valid_horizons}",
            )
        target_horizons = [request.horizon_minutes]
    else:
        target_horizons = valid_horizons

    weather_overrides = {}
    if request.temp is not None:
        weather_overrides["temp"] = request.temp
    if request.humidity is not None:
        weather_overrides["humidity"] = request.humidity
    if request.precip is not None:
        weather_overrides["precip"] = request.precip
    if request.windspeed is not None:
        weather_overrides["windspeed"] = request.windspeed

    forecasts = []
    for h in target_horizons:
        rec = forecasting_service.predict_zone(
            zone_id=request.zone_id,
            forecast_origin=origin_dt,
            horizon_minutes=h,
            weather_overrides=weather_overrides if weather_overrides else None,
            include_bounds=True,
        )
        forecasts.append(rec)

    # Compile recent historical demand leading up to origin (up to 12 hours)
    recent_history = []
    for offset in range(12, 0, -1):
        t_hist = origin_dt - pd.Timedelta(hours=offset)
        val = forecasting_service._get_historical_demand(request.zone_id, t_hist)
        recent_history.append({
            "timestamp": t_hist.strftime("%Y-%m-%dT%H:%M:%S"),
            "actual_rides": float(val),
        })
    val_origin = forecasting_service._get_historical_demand(request.zone_id, origin_dt)
    recent_history.append({
        "timestamp": origin_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "actual_rides": float(val_origin),
    })

    return {
        "zone_id": request.zone_id,
        "forecast_origin": origin_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "forecasts": forecasts,
        "recent_history": recent_history,
        "model_version": "layer1-multihorizon-v1.0",
        "target_description": "Passenger pickup ride demand (completed trips) per hourly interval.",
    }
