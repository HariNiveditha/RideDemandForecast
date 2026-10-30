"""Tests for Layer 3 integration contract schema, validation, export formats, and adapters."""

import json
from pathlib import Path
import pytest

from forecasting.layer3_contract import (
    ForecastRecord,
    Layer3Adapter,
    export_forecasts_to_csv,
    export_forecasts_to_json,
    get_layer3_json_schema,
    validate_forecast_records,
)


def test_forecast_record_valid():
    record_dict = {
        "zone_id": 161,
        "forecast_origin": "2026-10-28T10:00:00",
        "target_timestamp": "2026-10-28T11:00:00",
        "horizon_minutes": 60,
        "predicted_rides": 42.5,
        "lower_bound": 35.0,
        "upper_bound": 50.0,
        "model_version": "layer1-multihorizon-v1.0",
    }
    rec = ForecastRecord(**record_dict)
    assert rec.zone_id == 161
    assert rec.predicted_rides == 42.5
    assert rec.lower_bound == 35.0
    assert rec.upper_bound == 50.0


def test_forecast_record_invalid_bounds():
    # upper_bound cannot be less than lower_bound
    invalid_dict = {
        "zone_id": 161,
        "forecast_origin": "2026-10-28T10:00:00",
        "target_timestamp": "2026-10-28T11:00:00",
        "horizon_minutes": 60,
        "predicted_rides": 42.5,
        "lower_bound": 50.0,
        "upper_bound": 35.0,
    }
    with pytest.raises(ValueError):
        ForecastRecord(**invalid_dict)


def test_export_and_adapter(tmp_path):
    records = [
        {
            "zone_id": 161,
            "forecast_origin": "2021-06-15T12:00:00",
            "target_timestamp": "2021-06-15T13:00:00",
            "horizon_minutes": 60,
            "predicted_rides": 250.0,
            "lower_bound": 235.0,
            "upper_bound": 265.0,
        },
        {
            "zone_id": 237,
            "forecast_origin": "2021-06-15T12:00:00",
            "target_timestamp": "2021-06-15T13:00:00",
            "horizon_minutes": 60,
            "predicted_rides": 180.0,
            "lower_bound": 165.0,
            "upper_bound": 195.0,
        },
    ]

    # Validate
    validated = validate_forecast_records(records)
    assert len(validated) == 2

    # Export JSON
    json_path = tmp_path / "test_forecasts.json"
    export_forecasts_to_json(validated, json_path)
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded_json) == 2

    # Export CSV
    csv_path = tmp_path / "test_forecasts.csv"
    export_forecasts_to_csv(validated, csv_path)
    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "zone_id,forecast_origin,target_timestamp" in csv_content

    # Adapter to dataframe
    df = Layer3Adapter.to_dataframe(validated)
    assert len(df) == 2
    assert "predicted_rides" in df.columns

    # Adapter to zone matrix
    matrix = Layer3Adapter.to_zone_matrix(validated, horizon_minutes=60)
    assert matrix[161] == 250.0
    assert matrix[237] == 180.0


def test_layer3_schema_generation():
    schema = get_layer3_json_schema()
    assert "properties" in schema
    assert "zone_id" in schema["properties"]
    assert "predicted_rides" in schema["properties"]
    assert "target_timestamp" in schema["properties"]
