"""Tests for handling missing timestamps, quiet zones, and weather overrides."""

import pytest

from forecasting.service import ForecastingService


@pytest.fixture(scope="module")
def service():
    return ForecastingService.get_default_service()


def test_quiet_zone_with_sparse_history(service):
    # Zone 1 is Newark Airport (EWR), which has very sparse ride observations
    origin = "2021-06-15T03:00:00"  # 3 AM
    rec = service.predict_zone(zone_id=1, forecast_origin=origin, horizon_minutes=60)
    assert rec["zone_id"] == 1
    assert rec["predicted_rides"] >= 0.0
    assert rec["lower_bound"] >= 0.0
    assert rec["upper_bound"] >= rec["lower_bound"]


def test_weather_override_handling(service):
    origin = "2021-06-15T12:00:00"
    zone_id = 161

    # Standard weather
    base_rec = service.predict_zone(zone_id=zone_id, forecast_origin=origin, horizon_minutes=60)

    # Extreme blizzard weather override
    storm_weather = {
        "weather_temp": 20.0,
        "weather_snow": 15.0,
        "weather_windspeed": 35.0,
        "weather_precip": 1.5,
    }
    storm_rec = service.predict_zone(
        zone_id=zone_id,
        forecast_origin=origin,
        horizon_minutes=60,
        weather_overrides=storm_weather,
    )

    assert storm_rec["zone_id"] == 161
    assert storm_rec["predicted_rides"] >= 0.0
    assert isinstance(storm_rec["predicted_rides"], float)
