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
# Fleet simulation module
from fleet_simulation import (
    _make_initial_state,
    advance_simulation,
    compute_allocation_recommendations,
    DEMAND_PROPORTIONS,
    VALID_ZONE_IDS,
    NUM_ZONES,
    RANDOM_SEED,
    DEFAULT_FLEET_SIZE,
)

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
fleet_state = _make_initial_state()


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


from typing import Optional

# ============================================================================
# Fleet Management API Endpoints
# ============================================================================

class FleetInitRequest(BaseModel):
    fleet_size: Optional[int] = None
    seed: Optional[int] = None

@app.post("/fleet/init", response_model=None)
def fleet_init(
    req: Optional[FleetInitRequest] = None,
    fleet_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Initialize (or re-initialize) the fleet simulation state.

    - If a state already exists, it is replaced.
    - The state persists across requests.
    - Fleet uses historical demand proportions for zone distribution.
    """
    global fleet_state
    try:
        body_fs = req.fleet_size if req is not None else None
        body_seed = req.seed if req is not None else None
        fs = body_fs or fleet_size or DEFAULT_FLEET_SIZE
        sd = body_seed or seed or RANDOM_SEED
        fleet_state = _make_initial_state(fleet_size=fs, seed=sd)

        # Consistent demand prediction using historical-data interval calculation
        from fleet_simulation import _predict_demand_for_interval, SIMULATION_INTERVAL_HOURS
        fleet_state["predicted_rides"] = _predict_demand_for_interval(
            fleet_state, SIMULATION_INTERVAL_HOURS
        )

        return {
            "status": "initialized",
            "fleet_size": fleet_state["fleet_size"],
            "simulation_id": fleet_state["simulation_id"],
            "timestamp": fleet_state["timestamp"].isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fleet initialization error: {str(e)}")


@app.get("/fleet/current")
def fleet_current() -> dict:
    """
    Retrieve the current fleet state.

    Returns per-zone idle, busy, and repositioning cab counts,
    plus total fleet conservation check.
    """
    global fleet_state
    if fleet_state is None:
        raise HTTPException(status_code=503, detail="Fleet simulation state not initialized.")
    return {
        "simulation_id": fleet_state["simulation_id"],
        "timestamp": fleet_state["timestamp"].isoformat(),
        "fleet_size": fleet_state["fleet_size"],
        "predicted_rides": fleet_state.get("predicted_rides", {}),
        "availability": {
            zone: {
                "idle_cabs": fleet_state["idle_cabs"].get(zone, 0),
                "busy_cabs": fleet_state["busy_cabs"].get(zone, 0),
                "repositioning_cabs": fleet_state["repositioning_cabs"].get(zone, 0),
                "total_cabs": fleet_state["total_cabs_by_zone"].get(zone, 0),
            }
            for zone in fleet_state["zones"]
        },
        "fleet_totals": {
            "total_fleet": fleet_state["fleet_size"],
            "total_idle": sum(fleet_state["idle_cabs"].values()),
            "total_busy": sum(fleet_state["busy_cabs"].values()),
            "total_repositioning": sum(fleet_state["repositioning_cabs"].values()),
            "conservation_check": sum(fleet_state["total_cabs_by_zone"].values())
            == fleet_state["fleet_size"],
        },
    }


@app.post("/fleet/advance")
def fleet_advance() -> dict:
    """
    Advance the simulation by one interval.

    Models trip completions, repositioning, and new demand-based assignments.
    Uses the previous state (not independent regeneration).
    """
    global fleet_state
    if fleet_state is None:
        raise HTTPException(status_code=503, detail="Fleet simulation state not initialized.")
    fleet_state = advance_simulation(fleet_state)
    return {
        "status": "advanced",
        "timestamp": fleet_state["timestamp"].isoformat(),
        "conservation_check": sum(fleet_state["total_cabs_by_zone"].values()) == fleet_state["fleet_size"],
    }


@app.post("/fleet/reset")
def fleet_reset() -> dict:
    """
    Explicitly reset the simulation to the initial state.

    Preserves current configured fleet size.
    """
    global fleet_state
    current_size = (
        fleet_state.get("fleet_size", DEFAULT_FLEET_SIZE)
        if fleet_state
        else DEFAULT_FLEET_SIZE
    )
    fleet_state = _make_initial_state(fleet_size=current_size)
    from fleet_simulation import _predict_demand_for_interval, SIMULATION_INTERVAL_HOURS
    fleet_state["predicted_rides"] = _predict_demand_for_interval(
        fleet_state, SIMULATION_INTERVAL_HOURS
    )
    return {
        "status": "reset",
        "simulation_id": fleet_state["simulation_id"],
        "timestamp": fleet_state["timestamp"].isoformat(),
    }


@app.post("/fleet/recommendations")
def fleet_recommendations() -> dict:
    """
    Generate demand-aware cab allocation recommendations.

    Identifies zones with predicted shortages and surplus idle cabs,
    and recommends movements from donor to shortage zones.
    """
    global fleet_state
    if fleet_state is None:
        raise HTTPException(status_code=503, detail="Fleet simulation state not initialized.")
    rec = compute_allocation_recommendations(fleet_state)
    return rec


@app.get("/fleet/export")
def fleet_export() -> dict:
    """
    Export fleet snapshots as a list of per-zone records.

    Useful for reproducible evaluation and CSV/JSON export.
    """
    global fleet_state
    if fleet_state is None:
        raise HTTPException(status_code=503, detail="Fleet simulation state not initialized.")
    from fleet_simulation import _snapshot_for_export
    snapshots = _snapshot_for_export(fleet_state)
    return {
        "simulation_id": fleet_state["simulation_id"],
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }
