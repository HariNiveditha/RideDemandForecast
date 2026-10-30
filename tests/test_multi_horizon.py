"""Tests for multi-horizon model predictions across candidate horizons (60, 180, 360 min)."""

import numpy as np
import pytest

from forecasting.config import DEFAULT_HORIZONS_MINUTES
from forecasting.service import ForecastingService, forecast_batch, predict_zone


@pytest.fixture(scope="module")
def service():
    return ForecastingService.get_default_service()


def test_predict_multiple_horizons_distinct(service):
    # Verify that predictions are not identical across horizons
    origin = "2021-06-15T12:00:00"
    zone_id = 161  # Midtown Center

    preds = {}
    for h in DEFAULT_HORIZONS_MINUTES:
        rec = service.predict_zone(zone_id=zone_id, forecast_origin=origin, horizon_minutes=h)
        preds[h] = rec["predicted_rides"]
        assert rec["zone_id"] == zone_id
        assert rec["horizon_minutes"] == h
        assert rec["predicted_rides"] >= 0.0

    # Ensure forecasts for 1h, 3h, and 6h ahead are not merely copied
    unique_predictions = set(preds.values())
    assert len(unique_predictions) == len(DEFAULT_HORIZONS_MINUTES), (
        f"Forecasts must vary across horizons and not be copied: {preds}"
    )


def test_batch_forecasting_multiple_zones(service):
    zones = [161, 237, 132, 1]
    horizons = [60, 180, 360]
    origin = "2021-06-20T10:00:00"

    records = service.forecast_batch(
        zone_ids=zones,
        forecast_origin=origin,
        horizons_minutes=horizons,
        include_bounds=True,
    )

    expected_count = len(zones) * len(horizons)
    assert len(records) == expected_count

    for r in records:
        assert r["zone_id"] in zones
        assert r["horizon_minutes"] in horizons
        assert r["predicted_rides"] >= 0.0
        assert r["lower_bound"] is not None
        assert r["upper_bound"] is not None
        assert r["lower_bound"] >= 0.0
        assert r["upper_bound"] >= r["lower_bound"]
