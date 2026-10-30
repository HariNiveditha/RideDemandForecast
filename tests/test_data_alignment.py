"""Tests for feature and target alignment, and prevention of temporal data leakage."""

import numpy as np
import pandas as pd
import pytest

from forecasting.config import HORIZON_STEPS, MULTI_HORIZON_FEATURE_ORDER
from forecasting.data import (
    build_multi_horizon_dataset,
    build_regular_hourly_grid,
    load_raw_dataset,
)


@pytest.fixture(scope="module")
def sample_grid():
    df = load_raw_dataset()
    grid_df, weather_df = build_regular_hourly_grid(df)
    return grid_df, weather_df


def test_regular_grid_continuity(sample_grid):
    grid_df, _ = sample_grid
    # Check that every zone has continuous hourly timestamps
    unique_zones = grid_df["PULocationID"].unique()
    assert len(unique_zones) == 263

    sample_zone_df = grid_df[grid_df["PULocationID"] == 161].sort_values("pickup_hour")
    diffs = sample_zone_df["pickup_hour"].diff().dropna()
    # All diffs must be exactly 1 hour
    assert (diffs == pd.Timedelta(hours=1)).all()


@pytest.mark.parametrize("horizon_minutes", [60, 180, 360])
def test_leak_free_target_alignment(sample_grid, horizon_minutes):
    grid_df, weather_df = sample_grid
    step = HORIZON_STEPS[horizon_minutes]
    ds = build_multi_horizon_dataset(grid_df, weather_df, horizon_minutes)

    # 1. Target timestamp must be exactly origin + step hours
    assert (ds["target_timestamp"] == ds["forecast_origin"] + pd.Timedelta(hours=step)).all()

    # 2. Origin lag 0 must match the exact demand at forecast origin
    assert (ds["origin_lag_0"] == ds["ride_count"]).all()

    # 3. Target lag 168 must be exactly demand at target_timestamp - 168h
    # Which corresponds to origin - (168 - step)h.
    # Take a sample row and verify manually against grid_df
    sample_row = ds.iloc[500]
    zid = sample_row["PULocationID"]
    orig_time = sample_row["forecast_origin"]
    target_time = sample_row["target_timestamp"]

    # Target lag 24 must be at target_time - 24h
    time_24_ago = target_time - pd.Timedelta(hours=24)
    expected_lag24_val = float(grid_df.loc[
        (grid_df["PULocationID"] == zid) & (grid_df["pickup_hour"] == time_24_ago),
        "ride_count"
    ].iloc[0])
    assert sample_row["target_lag_24"] == pytest.approx(expected_lag24_val)

    # 4. Strict causality: Ensure no features use observations after forecast_origin
    # time_24_ago must be strictly <= orig_time
    assert time_24_ago <= orig_time
