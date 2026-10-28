"""Data ingestion, regularized hourly grid construction, and leak-free feature engineering."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting.config import (
    CALENDAR_FEATURES,
    DATA_PARQUET_PATH,
    DEFAULT_HORIZONS_MINUTES,
    HORIZON_STEPS,
    MULTI_HORIZON_FEATURE_ORDER,
    TAXI_ZONE_LOOKUP_PATH,
    TEST_CUTOFF,
    TEST_END,
    TRAIN_START,
    VAL_CUTOFF,
    WEATHER_FEATURES,
)

logger = logging.getLogger(__name__)


def load_raw_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the raw feature-engineered hourly parquet dataset."""
    target_path = path or DATA_PARQUET_PATH
    df = pd.read_parquet(target_path)
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"]).dt.floor("h")
    df["PULocationID"] = df["PULocationID"].astype(int)
    if "ride_count" in df.columns:
        df["ride_count"] = df["ride_count"].astype(float)
    return df


def load_zone_lookup(path: Optional[str] = None) -> pd.DataFrame:
    """Load taxi zone lookup table."""
    target_path = path or TAXI_ZONE_LOOKUP_PATH
    lookup = pd.read_csv(target_path)
    lookup = lookup.rename(columns={"LocationID": "PULocationID"})
    lookup["PULocationID"] = lookup["PULocationID"].astype(int)
    return lookup


def build_regular_hourly_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct a full regularized hourly grid for all active taxi zones.
    
    Missing hourly intervals for any taxi zone are filled with ride_count = 0.0,
    reflecting real periods with zero yellow taxi demand.
    City-wide weather variables are aligned and forward-filled if missing.
    
    Returns:
        grid_df: Complete dataframe indexed by [PULocationID, pickup_hour].
        weather_df: Unique hourly weather dataframe indexed by pickup_hour.
    """
    df = df.copy()
    min_hour = df["pickup_hour"].min()
    max_hour = df["pickup_hour"].max()
    full_hours = pd.date_range(min_hour, max_hour, freq="h", name="pickup_hour")
    unique_zones = np.sort(df["PULocationID"].unique())

    # Build unique hourly weather series
    weather_cols = [c for c in WEATHER_FEATURES if c in df.columns]
    weather_df = (
        df.groupby("pickup_hour")[weather_cols]
        .first()
        .reindex(full_hours)
        .ffill()
        .bfill()
    )

    # MultiIndex grid for all (zone, hour) pairs
    full_index = pd.MultiIndex.from_product(
        [unique_zones, full_hours], names=["PULocationID", "pickup_hour"]
    )

    # Index actual demand
    demand_series = (
        df.groupby(["PULocationID", "pickup_hour"])["ride_count"]
        .first()
        .reindex(full_index, fill_value=0.0)
    )

    grid_df = demand_series.to_frame()
    grid_df = grid_df.join(weather_df, on="pickup_hour").reset_index()

    return grid_df, weather_df


def build_multi_horizon_dataset(
    grid_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:
    """Build a leak-free dataset targeting horizon_minutes ahead.
    
    For forecast origin T:
        target_timestamp = T + step hours (e.g. T + 1h, T + 3h, T + 6h)
        target = ride_count at target_timestamp
        target_lag_168 = demand at target_timestamp - 168h = T - (168 - step)h <= T (strictly causal)
        target_lag_24 = demand at target_timestamp - 24h = T - (24 - step)h <= T (strictly causal)
        origin_lag_0 = demand at T
        origin_lag_1 = demand at T - 1h
        origin_rolling_mean_24 = rolling mean demand over [T-23h, T]
        origin_rolling_std_24 = rolling std demand over [T-23h, T]
        target calendar features = calendar attributes of target_timestamp
        target weather features = weather forecast at target_timestamp
    """
    step = HORIZON_STEPS[horizon_minutes]
    df = grid_df.sort_values(["PULocationID", "pickup_hour"]).reset_index(drop=True)

    grouped = df.groupby("PULocationID")["ride_count"]

    # Origin momentum features available as of pickup_hour (forecast origin T)
    df["origin_lag_0"] = df["ride_count"]
    df["origin_lag_1"] = grouped.shift(1)
    df["origin_rolling_mean_24"] = grouped.transform(
        lambda s: s.rolling(24, min_periods=1).mean()
    )
    df["origin_rolling_std_24"] = (
        grouped.transform(lambda s: s.rolling(24, min_periods=2).std()).fillna(0.0)
    )

    # Target-aligned historical lags:
    # Target time is T + step.
    # Same hour yesterday relative to target is: (T + step) - 24 = T - (24 - step).
    # Since step <= 6, 24 - step >= 18 > 0, so this is strictly in the past as of origin T!
    shift_target_lag24 = 24 - step
    shift_target_lag168 = 168 - step
    df["target_lag_24"] = grouped.shift(shift_target_lag24)
    df["target_lag_168"] = grouped.shift(shift_target_lag168)

    # Future target demand at T + step
    df["target"] = grouped.shift(-step)
    df["target_timestamp"] = df["pickup_hour"] + pd.Timedelta(hours=step)
    df["forecast_origin"] = df["pickup_hour"]
    df["horizon_minutes"] = horizon_minutes

    # Add target calendar features derived from target_timestamp
    target_dt = df["target_timestamp"].dt
    df["hour_of_day"] = target_dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dayofweek"] = target_dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["day"] = target_dt.day
    df["month"] = target_dt.month
    df["weekofyear"] = target_dt.isocalendar().week.astype(int)
    df["quarter"] = target_dt.quarter
    df["dayofyear"] = target_dt.dayofyear

    # Align target weather at target_timestamp
    weather_cols = [c for c in WEATHER_FEATURES if c in weather_df.columns]
    target_weather = weather_df[weather_cols].copy()
    target_weather.columns = [f"weather_{c}" for c in weather_cols]

    # Drop old weather columns
    existing_raw_weather = [c for c in WEATHER_FEATURES if c in df.columns]
    df = df.drop(columns=existing_raw_weather, errors="ignore")

    df = df.join(target_weather, on="target_timestamp")

    # Drop initial rows without sufficient lag history or without future target
    required_cols = ["target", "target_lag_168", "target_lag_24", "origin_lag_1"]
    df = df.dropna(subset=required_cols).copy()

    return df


def split_chronological(
    df: pd.DataFrame,
    val_cutoff: str = VAL_CUTOFF,
    test_cutoff: str = TEST_CUTOFF,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict chronological split by forecast_origin timestamp.
    
    Train: forecast_origin < val_cutoff
    Val:   val_cutoff <= forecast_origin < test_cutoff
    Test:  forecast_origin >= test_cutoff
    """
    val_dt = pd.Timestamp(val_cutoff)
    test_dt = pd.Timestamp(test_cutoff)

    train_df = df[df["forecast_origin"] < val_dt].copy()
    val_df = df[(df["forecast_origin"] >= val_dt) & (df["forecast_origin"] < test_dt)].copy()
    test_df = df[df["forecast_origin"] >= test_dt].copy()

    return train_df, val_df, test_df
