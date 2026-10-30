"""Tests ensuring that existing application workflows and inference endpoints remain non-broken."""

import joblib
import pytest

import main


@pytest.fixture(scope="module")
def setup_main():
    main.model = joblib.load("rf_model.pkl")
    main.history = main._load_history()
    main.history_by_key = main.history["ride_count"]
    return main


def test_existing_metrics_endpoint(setup_main):
    m = setup_main.get_metrics()
    assert "rmse" in m
    assert "mae" in m
    assert "r2" in m
    assert "feature_importances" in m
    assert len(m["feature_importances"]) > 0


def test_existing_locations_endpoint(setup_main):
    locs = setup_main.get_locations()
    assert "locations" in locs
    assert len(locs["locations"]) == 263
    first_loc = locs["locations"][0]
    assert "PULocationID" in first_loc
    assert "Borough" in first_loc
    assert "Zone" in first_loc


def test_existing_predictions_sample_endpoint(setup_main):
    sample = setup_main.get_predictions_sample()
    assert "points" in sample
    assert len(sample["points"]) > 0
    first_point = sample["points"][0]
    assert "timestamp" in first_point
    assert "actual" in first_point
    assert "predicted" in first_point


def test_existing_predict_request(setup_main):
    req = setup_main.PredictionRequest(
        PULocationID=161,
        datetime="2021-06-15T12:00:00",
        temp=72.0,
        feelslike=72.0,
        humidity=50.0,
        precip=0.0,
        precipprob=0.0,
        snow=0.0,
        windgust=5.0,
        windspeed=8.0,
        sealevelpressure=1013.0,
        cloudcover=10.0,
    )
    result = setup_main.predict(req)
    assert "predicted_demand" in result
    assert isinstance(result["predicted_demand"], float)
    assert result["predicted_demand"] > 0.0
