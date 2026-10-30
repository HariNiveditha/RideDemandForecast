"""Tests for forecast data-freshness disclosure and timestamp handling.

The project has no live demand feed, so forecasts are anchored to the newest
hour in the historical dataset (2021-06-30T23:00:00), never to the system
clock. These tests lock down:

  * the dataset timestamp and the system timestamp are reported separately,
  * the default forecast origin comes from the data, not from "now",
  * 60/180/360-minute horizons stay measured from the chosen origin,
  * dataset-anchored data is never described as live,
  * an origin in the future relative to the data is rejected (no fabrication),
  * the historical default and the explicit fallback contract are unchanged.

See FORECAST_DATA_FRESHNESS.md for the design rationale.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from fleet_simulation import _make_initial_state

FLEET_SIZE = 20000
HISTORY_PATH = main.HISTORY_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def deterministic_fleet():
    main.fleet_state = _make_initial_state(fleet_size=FLEET_SIZE, seed=42)
    yield
    main.fleet_state = _make_initial_state(fleet_size=FLEET_SIZE, seed=42)


def _dataset_latest_from_parquet() -> str:
    """Independently read the newest dataset hour straight from the parquet."""
    frame = pd.read_parquet(HISTORY_PATH, columns=["pickup_hour"])
    return pd.Timestamp(pd.to_datetime(frame["pickup_hour"]).max()).floor("h").strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


# ---------------------------------------------------------------------------
# 1. Dataset time vs system time are reported separately
# ---------------------------------------------------------------------------


def test_freshness_reports_dataset_and_system_time_separately(client):
    response = client.post("/fleet/recommendations")
    assert response.status_code == 200
    source = response.json()["demand_source"]

    expected_latest = _dataset_latest_from_parquet()
    assert source["dataset_latest_timestamp"] == expected_latest
    assert source["system_time"] is not None
    assert source["data_basis"] == "historical_dataset"
    assert source["is_live"] is False

    # The whole point: these two must be distinct concepts.
    assert source["dataset_latest_timestamp"] != source["system_time"]
    assert expected_latest != source["system_time"]

    # The system clock is far ahead of the data -- proving it was not used as
    # the demand/forecast timestamp.
    assert pd.Timestamp(source["system_time"].rstrip("Z")) > pd.Timestamp(expected_latest)


def test_freshness_note_states_no_live_feed(client):
    for body in (None, {"horizon_minutes": 60}, {"horizon_minutes": 60, "allow_fallback": True}):
        response = (
            client.post("/fleet/recommendations")
            if body is None
            else client.post("/fleet/recommendations", json=body)
        )
        assert response.status_code == 200, response.text
        source = response.json()["demand_source"]
        note = source["freshness_note"]
        assert "No live demand feed is configured" in note
        assert source["dataset_latest_timestamp"] in note
        assert source["system_time"] in note
        # Never claims to be live.
        assert "is not live" in note or "not to the current system time" in note


def test_historical_default_unchanged_by_freshness_work(client):
    """Adding freshness fields must not alter the historical default path."""
    response = client.post("/fleet/recommendations")
    assert response.status_code == 200
    payload = response.json()
    assert payload["demand_source"]["type"] == "historical"
    assert payload["demand_source"]["zones_forecasted"] == 0
    # Original fields are all still present.
    for key in (
        "simulation_id",
        "prediction_interval",
        "zones",
        "recommended_movements",
        "unmet_demand_estimate",
        "demand_source",
    ):
        assert key in payload
    # Legacy bodyless/empty-body calls still work.
    for body in ({}, {"horizon_minutes": None}):
        assert (
            client.post("/fleet/recommendations", json=body).json()["demand_source"]["type"]
            == "historical"
        )


# ---------------------------------------------------------------------------
# 2. Default origin comes from the data, never from the clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [60, 180, 360])
def test_default_origin_is_dataset_hour_not_current_time(client, horizon):
    response = client.post("/fleet/recommendations", json={"horizon_minutes": horizon})
    assert response.status_code == 200, response.text
    source = response.json()["demand_source"]

    dataset_latest = _dataset_latest_from_parquet()
    assert source["forecast_origin"] == dataset_latest
    assert source["origin_is_dataset_latest"] is True

    origin = pd.Timestamp(source["forecast_origin"])
    # Anchored to the dataset's year, not to the current system clock.
    assert origin.year == 2021
    assert origin != pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))

    assert source["is_live"] is False
    assert source["data_basis"] == "historical_dataset"


@pytest.mark.parametrize("horizon", [60, 180, 360])
def test_horizon_is_measured_from_origin(client, horizon):
    """Horizon semantics stay 'origin + N minutes' for every supported value."""
    response = client.post("/fleet/recommendations", json={"horizon_minutes": horizon})
    assert response.status_code == 200
    source = response.json()["demand_source"]
    origin = pd.Timestamp(source["forecast_origin"])
    target = pd.Timestamp(source["target_timestamp"])
    assert target == origin + pd.Timedelta(minutes=horizon)
    assert source["horizon_minutes"] == horizon
    assert response.json()["prediction_interval"] == "1 hour"


def test_horizons_target_distinct_hours(client):
    targets = set()
    origins = set()
    for horizon in (60, 180, 360):
        source = client.post(
            "/fleet/recommendations", json={"horizon_minutes": horizon}
        ).json()["demand_source"]
        targets.add(source["target_timestamp"])
        origins.add(source["forecast_origin"])
    assert len(origins) == 1
    assert len(targets) == 3


def test_target_beyond_dataset_end_flagged(client):
    dataset_latest = pd.Timestamp(_dataset_latest_from_parquet())

    # Default origin = dataset end, so any horizon lands past the data.
    source = client.post("/fleet/recommendations", json={"horizon_minutes": 60}).json()[
        "demand_source"
    ]
    assert source["target_beyond_dataset_end"] is True

    # An earlier in-range origin targets an observed hour.
    source = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": "2021-06-30T12:00:00"},
    ).json()["demand_source"]
    assert source["origin_is_dataset_latest"] is False
    assert source["target_beyond_dataset_end"] is False
    assert pd.Timestamp(source["target_timestamp"]) <= dataset_latest


def test_system_time_origin_rejected_no_fabrication(client):
    """Asking for a 'now' origin must not silently produce a fabricated forecast."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": now.isoformat()},
    )
    assert response.status_code == 400
    assert "after the newest available data hour" in response.json()["detail"]


def test_far_future_origin_rejected(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": "2030-01-01T00:00:00"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 3. Fallback keeps its contract and carries the same disclosure
# ---------------------------------------------------------------------------


class _BrokenService:
    grid_df = None

    def forecast_batch(self, **kwargs):
        raise RuntimeError("simulated outage")


def test_fallback_carries_freshness_disclosure(client, monkeypatch):
    monkeypatch.setattr(main, "forecasting_service", _BrokenService())
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "allow_fallback": True},
    )
    assert response.status_code == 200
    source = response.json()["demand_source"]
    assert source["type"] == "historical_fallback"
    assert source["forecast_error"] is not None
    assert source["data_basis"] == "historical_dataset"
    assert source["is_live"] is False
    assert source["dataset_latest_timestamp"] == _dataset_latest_from_parquet()
    assert source["system_time"] is not None

    # Without allow_fallback the failure is still an error, never a silent swap.
    monkeypatch.setattr(main, "forecasting_service", _BrokenService())
    assert (
        client.post("/fleet/recommendations", json={"horizon_minutes": 60}).status_code == 503
    )


# ---------------------------------------------------------------------------
# 4. Forecasting endpoints disclose the same freshness metadata
# ---------------------------------------------------------------------------


def test_horizons_endpoint_reports_freshness(client):
    data = client.get("/forecast/horizons").json()
    # Existing contract preserved.
    assert [h["minutes"] for h in data["supported_horizons"]] == [60, 180, 360]
    assert data["has_uncertainty_bounds"] is True
    # New freshness disclosure.
    assert data["data_basis"] == "historical_dataset"
    assert data["is_live"] is False
    assert data["dataset_latest_timestamp"] == _dataset_latest_from_parquet()
    assert data["system_time"] is not None
    assert "dataset's latest available hour" in data["horizon_reference"]


def test_multi_horizon_endpoint_reports_freshness(client):
    response = client.post(
        "/forecast/multi-horizon",
        json={"zone_id": 161, "forecast_origin": "2021-06-15T12:00:00"},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    # Existing contract preserved.
    assert data["forecast_origin"] == "2021-06-15T12:00:00"
    assert len(data["forecasts"]) == 3
    assert len(data["recent_history"]) > 0

    # Freshness disclosure.
    assert data["data_basis"] == "historical_dataset"
    assert data["is_live"] is False
    assert data["dataset_latest_timestamp"] == _dataset_latest_from_parquet()
    assert data["system_time"] is not None
    assert data["origin_is_dataset_latest"] is False
    assert data["origin_within_dataset_range"] is True

    # Horizons remain origin-relative here too.
    for record in data["forecasts"]:
        expected = pd.Timestamp("2021-06-15T12:00:00") + pd.Timedelta(
            minutes=record["horizon_minutes"]
        )
        assert pd.Timestamp(record["target_timestamp"]) == expected


def test_multi_horizon_default_origin_not_current_time(client):
    """A dataset-anchored origin must never be replaced by the system clock."""
    response = client.post(
        "/forecast/multi-horizon",
        json={"zone_id": 161, "forecast_origin": _dataset_latest_from_parquet()},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["origin_is_dataset_latest"] is True
    assert data["forecast_origin"].startswith("2021-")
    assert pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)).year != int(
        data["forecast_origin"][:4]
    )
