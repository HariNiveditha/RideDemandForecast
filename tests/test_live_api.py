"""End-to-end API integration tests for FastAPI application using TestClient."""

from fastapi.testclient import TestClient
import pytest

import main


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


def test_docs_and_openapi(client):
    r_docs = client.get("/docs")
    assert r_docs.status_code == 200

    r_openapi = client.get("/openapi.json")
    assert r_openapi.status_code == 200
    schema = r_openapi.json()
    paths = schema.get("paths", {})

    assert "/forecast/multi-horizon" in paths
    assert "/forecast/horizons" in paths
    assert "/predict" in paths
    assert "/metrics" in paths
    assert "/locations" in paths


def test_forecast_horizons_endpoint(client):
    res = client.get("/forecast/horizons")
    assert res.status_code == 200
    data = res.json()
    assert "supported_horizons" in data
    assert len(data["supported_horizons"]) == 3
    minutes = [h["minutes"] for h in data["supported_horizons"]]
    assert minutes == [60, 180, 360]
    assert data["has_uncertainty_bounds"] is True


def test_multi_horizon_endpoint_all_horizons(client):
    payload = {
        "zone_id": 161,
        "forecast_origin": "2021-06-15T12:00:00",
        "horizon_minutes": None,
        "temp": 72.0,
        "humidity": 50.0,
        "precip": 0.0,
    }
    res = client.post("/forecast/multi-horizon", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["zone_id"] == 161
    assert data["forecast_origin"] == "2021-06-15T12:00:00"
    assert len(data["forecasts"]) == 3

    h_map = {f["horizon_minutes"]: f for f in data["forecasts"]}
    assert 60 in h_map
    assert 180 in h_map
    assert 360 in h_map

    # Ensure predictions vary across horizons and bounds are valid
    for h, f in h_map.items():
        assert f["predicted_rides"] >= 0.0
        assert f["lower_bound"] is not None
        assert f["upper_bound"] is not None
        assert f["lower_bound"] >= 0.0
        assert f["upper_bound"] >= f["lower_bound"]
        assert f["target_timestamp"] > data["forecast_origin"]

    assert len(data["recent_history"]) > 0
    first_hist = data["recent_history"][0]
    assert "timestamp" in first_hist
    assert "actual_rides" in first_hist


def test_multi_horizon_endpoint_single_horizon(client):
    payload = {
        "zone_id": 237,
        "forecast_origin": "2021-06-15T14:00:00",
        "horizon_minutes": 180,
    }
    res = client.post("/forecast/multi-horizon", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["forecasts"]) == 1
    f = data["forecasts"][0]
    assert f["horizon_minutes"] == 180
    assert f["target_timestamp"] == "2021-06-15T17:00:00"


def test_multi_horizon_invalid_horizon_fails(client):
    payload = {
        "zone_id": 161,
        "forecast_origin": "2021-06-15T12:00:00",
        "horizon_minutes": 999,
    }
    res = client.post("/forecast/multi-horizon", json=payload)
    assert res.status_code == 400


def test_legacy_predict_preserved(client):
    payload = {
        "PULocationID": 161,
        "datetime": "2021-06-15T12:00:00",
        "temp": 72.0,
        "feelslike": 72.0,
        "humidity": 50.0,
        "precip": 0.0,
        "precipprob": 0.0,
        "snow": 0.0,
        "windgust": 5.0,
        "windspeed": 8.0,
        "sealevelpressure": 1013.0,
        "cloudcover": 10.0,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_demand" in data
    assert isinstance(data["predicted_demand"], float)


def test_layer2_anomalies_endpoint(client):
    res = client.get("/anomalies?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "anomalies" in data
    assert len(data["anomalies"]) <= 10
    if data["anomalies"]:
        rec = data["anomalies"][0]
        assert "zone_id" in rec
        assert "anomaly_type" in rec
        assert "severity" in rec
        assert "explanation" in rec


def test_layer2_anomalies_summary_endpoint(client):
    res = client.get("/anomalies/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_anomalies" in data
    assert "by_type" in data
    assert "by_severity" in data
    assert "top_zones" in data
    assert data["total_anomalies"] > 0


def test_layer2_forecast_performance_endpoint(client):
    res = client.get("/forecast-performance")
    assert res.status_code == 200
    data = res.json()
    assert "horizons" in data
    assert len(data["horizons"]) == 3
    assert data["error_convention"] == "error = actual - predicted (positive = underprediction, negative = overprediction)"


def test_layer2_forecast_errors_zones_endpoint(client):
    res = client.get("/forecast-errors/zones?horizon=60&sort_by=mae&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["horizon_minutes"] == 60
    assert "zones" in data
    assert len(data["zones"]) <= 5


def test_layer2_forecast_errors_timeseries_endpoint(client):
    res = client.get("/forecast-errors/timeseries?horizon=60")
    assert res.status_code == 200
    data = res.json()
    assert data["horizon_minutes"] == 60
    assert "timeseries" in data
    assert len(data["timeseries"]) == 30

