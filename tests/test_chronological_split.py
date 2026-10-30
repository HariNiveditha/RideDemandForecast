"""Tests for strict chronological train/validation/test separation without temporal overlap."""

import pandas as pd
import pytest

from forecasting.config import TEST_CUTOFF, VAL_CUTOFF
from forecasting.data import (
    build_multi_horizon_dataset,
    build_regular_hourly_grid,
    load_raw_dataset,
    split_chronological,
)


@pytest.fixture(scope="module")
def sample_dataset():
    df = load_raw_dataset()
    grid_df, weather_df = build_regular_hourly_grid(df)
    return build_multi_horizon_dataset(grid_df, weather_df, 60)


def test_chronological_separation(sample_dataset):
    train_df, val_df, test_df = split_chronological(sample_dataset)

    # 1. Non-empty splits
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    # 2. Strict chronological order of forecast origin timestamps
    max_train_origin = train_df["forecast_origin"].max()
    min_val_origin = val_df["forecast_origin"].min()
    max_val_origin = val_df["forecast_origin"].max()
    min_test_origin = test_df["forecast_origin"].min()

    assert max_train_origin < min_val_origin
    assert max_val_origin < min_test_origin

    # 3. Exact alignment with cutoffs
    assert max_train_origin < pd.Timestamp(VAL_CUTOFF)
    assert min_val_origin >= pd.Timestamp(VAL_CUTOFF)
    assert max_val_origin < pd.Timestamp(TEST_CUTOFF)
    assert min_test_origin >= pd.Timestamp(TEST_CUTOFF)

    # 4. Zero overlap in origin timestamps
    train_origins = set(train_df["forecast_origin"].unique())
    val_origins = set(val_df["forecast_origin"].unique())
    test_origins = set(test_df["forecast_origin"].unique())

    assert len(train_origins.intersection(val_origins)) == 0
    assert len(val_origins.intersection(test_origins)) == 0
    assert len(train_origins.intersection(test_origins)) == 0
