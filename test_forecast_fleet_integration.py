"""Integration tests: Layer 1 forecasts driving Layer 3 fleet recommendations.

These tests exercise the real code path end to end:

    FastAPI endpoint -> ForecastingService.forecast_batch()
                     -> Layer3Adapter.to_fleet_demand()  (Layer 3 contract)
                     -> compute_allocation_recommendations()

Positive-path tests use the real forecasting service and trained artefacts.
Negative-path tests replace only ``ForecastingService.forecast_batch`` (the
upstream producer) so that the Layer 3 adapter and the endpoint validation
logic still run for real.

Covers:
  * forecast-driven recommendations for the 60 / 180 / 360 minute horizons
  * historical-demand fallback when no horizon is requested
  * explicit fallback contract when forecasting fails (allow_fallback)
  * rejection of unsupported horizons, invalid zones, negative / non-finite
    demand, duplicate and missing records, and stale forecast origins
  * fleet conservation and state immutability around forecast recommendations
"""

import copy
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from fleet_simulation import (
    _make_initial_state,
    advance_simulation,
    compute_allocation_recommendations,
    _predict_demand_for_interval,
    SIMULATION_INTERVAL_HOURS,
    VALID_ZONE_IDS,
)
from forecasting.layer3_contract import (
    ForecastContractError,
    Layer3Adapter,
)

FLEET_SIZE = 20000
LATEST_DATA_HOUR = "2021-06-30 23:00:00"
SAMPLE_ZONES = [61, 79, 104, 132, 137]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def deterministic_fleet():
    """Give every test an identical, known fleet state."""
    main.fleet_state = _make_initial_state(fleet_size=FLEET_SIZE, seed=42)
    yield
    main.fleet_state = _make_initial_state(fleet_size=FLEET_SIZE, seed=42)


@pytest.fixture(autouse=True)
def forecasting_service_slot():
    """Restore the module-level forecasting service after each test."""
    original = main.forecasting_service
    yield
    main.forecasting_service = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(
    zone_ids,
    horizon_minutes=60,
    origin="2021-06-30T23:00:00",
    demand=10.0,
    override=None,
):
    """Build well-formed Layer 1 forecast records for stubbed service replies."""
    origin_ts = pd.Timestamp(origin)
    target_ts = origin_ts + pd.Timedelta(minutes=horizon_minutes)
    records = []
    for zone_id in zone_ids:
        record = {
            "zone_id": int(zone_id),
            "forecast_origin": origin_ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_timestamp": target_ts.strftime("%Y-%m-%dT%H:%M:%S"),
            "horizon_minutes": int(horizon_minutes),
            "predicted_rides": float(demand),
            "lower_bound": 0.0,
            "upper_bound": max(1.0, float(demand) * 1.2),
            "model_version": "layer1-multihorizon-v1.0",
        }
        if override:
            record.update(override(zone_id) if callable(override) else override)
        records.append(record)
    return records


class _StubForecastingService:
    """Stand-in exposing only what the fleet recommendations endpoint uses.

    Replaces the upstream forecast producer so that the Layer 3 adapter and
    the endpoint's validation/fallback logic still execute for real.
    """

    def __init__(self, records=None, error=None, grid_latest=LATEST_DATA_HOUR):
        self._records = records if records is not None else []
        self._error = error
        if grid_latest is None:
            self.grid_df = None
        else:
            self.grid_df = pd.DataFrame({"pickup_hour": [pd.Timestamp(grid_latest)]})
        self.calls = []

    def forecast_batch(self, zone_ids=None, forecast_origin=None, horizons_minutes=None, **kwargs):
        self.calls.append(
            {
                "zone_ids": list(zone_ids or []),
                "forecast_origin": forecast_origin,
                "horizons_minutes": list(horizons_minutes or []),
            }
        )
        if self._error is not None:
            raise self._error
        return [dict(record) for record in self._records]


def _install_service(**kwargs):
    stub = _StubForecastingService(**kwargs)
    main.forecasting_service = stub
    return stub


def _zone_predictions(response_payload):
    return {zone["zone_id"]: zone["predicted_rides"] for zone in response_payload["zones"]}


def _assert_allocation_invariants(payload):
    """Allocation rules that must hold for every demand source."""
    zones = {zone["zone_id"]: zone for zone in payload["zones"]}
    donor_donated = {}
    for move in payload["recommended_movements"]:
        assert move["from_zone"] != move["to_zone"], f"self movement: {move}"
        assert 1 <= move["cab_count"] <= 5, f"cab count out of bounds: {move}"
        assert zones[move["from_zone"]]["surplus_after_own_demand"] > 0
        assert zones[move["to_zone"]]["shortage_estimate"] > 0
        donor_donated[move["from_zone"]] = (
            donor_donated.get(move["from_zone"], 0) + move["cab_count"]
        )
    for donor_id, total in donor_donated.items():
        assert total <= zones[donor_id]["surplus_after_own_demand"], (
            f"donor {donor_id} over-donated {total} > "
            f"{zones[donor_id]['surplus_after_own_demand']}"
        )
    for zone in payload["zones"]:
        assert zone["shortage_estimate"] >= 0
        assert zone["surplus"] >= 0
        assert zone["predicted_rides"] >= 0


def _assert_fleet_conserved():
    current = main.fleet_state
    assert sum(current["idle_cabs"].values()) + sum(current["busy_cabs"].values()) + \
        sum(current["repositioning_cabs"].values()) == current["fleet_size"]
    assert sum(current["total_cabs_by_zone"].values()) == current["fleet_size"]
    for zone_id in current["zones"]:
        expected = (
            current["idle_cabs"].get(zone_id, 0)
            + current["busy_cabs"].get(zone_id, 0)
            + current["repositioning_cabs"].get(zone_id, 0)
        )
        assert current["total_cabs_by_zone"][zone_id] == expected
        assert current["idle_cabs"][zone_id] >= 0
        assert current["busy_cabs"][zone_id] >= 0
        assert current["repositioning_cabs"][zone_id] >= 0


# ---------------------------------------------------------------------------
# 1. Forecast-driven recommendations (60 / 180 / 360 minutes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [60, 180, 360])
def test_forecast_horizon_drives_recommendations(client, horizon):
    response = client.post("/fleet/recommendations", json={"horizon_minutes": horizon})
    assert response.status_code == 200, response.text
    payload = response.json()

    source = payload["demand_source"]
    assert source["type"] == "forecast"
    assert source["requested_horizon_minutes"] == horizon
    assert source["horizon_minutes"] == horizon
    assert source["forecast_origin"] is not None
    assert source["target_timestamp"] is not None
    assert source["zones_forecasted"] == len(VALID_ZONE_IDS)
    assert source["zones_from_history"] == 0
    assert source["forecast_error"] is None

    # Forecast demand must differ from the historical baseline for at least one zone.
    historical = _predict_demand_for_interval(main.fleet_state, SIMULATION_INTERVAL_HOURS)
    predictions = _zone_predictions(payload)
    assert any(
        predictions[zone_id] != historical[zone_id] for zone_id in VALID_ZONE_IDS
    ), "forecast demand was identical to historical demand"

    # Response keeps the original schema.
    for key in (
        "simulation_id",
        "prediction_interval",
        "zones",
        "recommended_movements",
        "unmet_demand_estimate",
    ):
        assert key in payload
    assert payload["simulation_id"] == main.fleet_state["simulation_id"]
    _assert_allocation_invariants(payload)


def test_horizons_map_to_distinct_target_intervals(client):
    """Each horizon forecasts a different target hour but the same 1-hour demand unit."""
    targets = {}
    for horizon in (60, 180, 360):
        response = client.post("/fleet/recommendations", json={"horizon_minutes": horizon})
        assert response.status_code == 200, response.text
        source = response.json()["demand_source"]
        origin = pd.Timestamp(source["forecast_origin"])
        target = pd.Timestamp(source["target_timestamp"])
        assert target == origin + pd.Timedelta(minutes=horizon)
        # predicted_rides is interval demand for one hour, so the interval length
        # implied by the response metadata must stay 1 hour.
        assert response.json()["prediction_interval"] == "1 hour"
        targets[horizon] = (source["forecast_origin"], source["target_timestamp"], source["total_predicted_rides"])

    assert len({t[1] for t in targets.values()}) == 3, "horizons must target distinct hours"


def test_forecast_demand_matches_layer3_adapter_output(client):
    """The endpoint's demand equals Layer3Adapter output from the real service."""
    from forecasting.service import ForecastingService

    service = ForecastingService.get_default_service()
    origin = pd.Timestamp(service.grid_df["pickup_hour"].max()).floor("h")
    latest = origin

    records = service.forecast_batch(
        zone_ids=SAMPLE_ZONES,
        forecast_origin=origin,
        horizons_minutes=[60],
    )
    expected, meta = Layer3Adapter.to_fleet_demand(
        records,
        60,
        SAMPLE_ZONES,
        expected_origin=origin,
        latest_available_timestamp=latest,
    )
    assert meta["horizon_minutes"] == 60

    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": SAMPLE_ZONES},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    predictions = _zone_predictions(payload)

    for zone_id in SAMPLE_ZONES:
        assert predictions[zone_id] == pytest.approx(expected[zone_id]), (
            f"zone {zone_id}: endpoint demand {predictions[zone_id]} != "
            f"adapter output {expected[zone_id]}"
        )

    source = payload["demand_source"]
    assert source["type"] == "forecast"
    assert source["zones_forecasted"] == len(SAMPLE_ZONES)
    assert source["zones_from_history"] == len(VALID_ZONE_IDS) - len(SAMPLE_ZONES)

    # Zones outside the requested subset keep historical demand.
    historical = _predict_demand_for_interval(main.fleet_state, SIMULATION_INTERVAL_HOURS)
    for zone_id in set(VALID_ZONE_IDS) - set(SAMPLE_ZONES):
        assert predictions[zone_id] == historical[zone_id]


def test_forecast_recommendations_do_not_mutate_state(client):
    before = copy.deepcopy(main.fleet_state)
    first = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    second = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    assert first.status_code == 200 and second.status_code == 200
    assert main.fleet_state == before, "forecast recommendations mutated the simulation state"

    # `demand_source.system_time` is deliberately a per-request clock reading: it
    # is the disclosure that separates "now" from the dataset's latest hour (see
    # FORECAST_DATA_FRESHNESS.md). It is the ONE field allowed to vary between two
    # identical requests -- everything else must still match exactly, so any other
    # nondeterminism still fails here.
    first_payload = first.json()
    second_payload = second.json()
    first_source = dict(first_payload.pop("demand_source"))
    second_source = dict(second_payload.pop("demand_source"))
    first_clock = first_source.pop("system_time")
    second_clock = second_source.pop("system_time")

    assert first_source == second_source, "demand/freshness metadata disagreed beyond system_time"
    assert first_payload == second_payload, "repeated identical requests disagreed"

    # The clock reading must be a real recent timestamp sitting after the dataset
    # hour, which is what proves it was reported rather than used as demand time.
    first_ts = pd.Timestamp(first_clock.rstrip("Z"))
    second_ts = pd.Timestamp(second_clock.rstrip("Z"))
    assert abs((second_ts - first_ts).total_seconds()) < 60
    assert first_ts > pd.Timestamp(main._dataset_latest_timestamp())


def test_forecast_recommendations_preserve_conservation(client):
    init = client.post("/fleet/init", json={"fleet_size": 13000})
    assert init.status_code == 200
    _assert_fleet_conserved()

    for step in range(1, 4):
        recs = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
        assert recs.status_code == 200, recs.text
        _assert_allocation_invariants(recs.json())
        _assert_fleet_conserved()

        advance = client.post("/fleet/advance")
        assert advance.status_code == 200
        assert advance.json()["conservation_check"] is True
        _assert_fleet_conserved()

        current = client.get("/fleet/current").json()
        assert current["fleet_totals"]["conservation_check"] is True
        assert (
            current["fleet_totals"]["total_idle"]
            + current["fleet_totals"]["total_busy"]
            + current["fleet_totals"]["total_repositioning"]
            == 13000
        )

    # Repeated advances keep working with forecast-driven recommendations.
    assert step == 3


def test_forecast_driven_and_historical_share_allocation_maths(client):
    """Same state, different demand inputs -> both valid, both documented."""
    historical = client.post("/fleet/recommendations")
    assert historical.status_code == 200
    forecast = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    assert forecast.status_code == 200

    hist_payload, fc_payload = historical.json(), forecast.json()
    assert hist_payload["demand_source"]["type"] == "historical"
    assert fc_payload["demand_source"]["type"] == "forecast"
    assert hist_payload["prediction_interval"] == fc_payload["prediction_interval"]
    _assert_allocation_invariants(hist_payload)
    _assert_allocation_invariants(fc_payload)
    # Zone metric columns are identical between both sources.
    assert set(hist_payload["zones"][0].keys()) == set(fc_payload["zones"][0].keys())


# ---------------------------------------------------------------------------
# 2. Historical fallback behaviour
# ---------------------------------------------------------------------------


def test_no_request_body_uses_historical_demand(client):
    response = client.post("/fleet/recommendations")
    assert response.status_code == 200
    payload = response.json()

    source = payload["demand_source"]
    assert source["type"] == "historical"
    assert source["horizon_minutes"] is None
    assert source["zones_forecasted"] == 0
    assert source["zones_from_history"] == len(VALID_ZONE_IDS)
    assert source["forecast_error"] is None

    historical = _predict_demand_for_interval(main.fleet_state, SIMULATION_INTERVAL_HOURS)
    predictions = _zone_predictions(payload)
    for zone_id in VALID_ZONE_IDS:
        assert predictions[zone_id] == historical[zone_id]
    _assert_allocation_invariants(payload)


def test_empty_body_uses_historical_demand(client):
    for body in ({}, {"horizon_minutes": None}, {"simulation_id": "sim-legacy"}):
        response = client.post("/fleet/recommendations", json=body)
        assert response.status_code == 200, f"{body}: {response.text}"
        assert response.json()["demand_source"]["type"] == "historical"


def test_no_horizon_with_zone_ids_still_uses_historical(client):
    response = client.post("/fleet/recommendations", json={"zone_ids": SAMPLE_ZONES})
    assert response.status_code == 200
    assert response.json()["demand_source"]["type"] == "historical"


def test_forecasting_service_unavailable_returns_error(client):
    _install_service(error=RuntimeError("simulated outage"))
    response = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    assert response.status_code == 503
    assert "simulated outage" in response.json()["detail"]
    assert response.json()["detail"].startswith("Forecasting service unavailable")

    # The default contract must not silently claim forecast-driven allocation.
    assert "demand_source" not in response.json()


def test_forecasting_service_unavailable_fallback_when_explicitly_allowed(client):
    _install_service(error=RuntimeError("simulated outage"))
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "allow_fallback": True},
    )
    assert response.status_code == 200
    payload = response.json()

    source = payload["demand_source"]
    assert source["type"] == "historical_fallback"
    assert source["requested_horizon_minutes"] == 60
    assert source["forecast_error"] is not None
    assert "simulated outage" in source["forecast_error"]

    historical = _predict_demand_for_interval(main.fleet_state, SIMULATION_INTERVAL_HOURS)
    for zone_id in VALID_ZONE_IDS:
        assert _zone_predictions(payload)[zone_id] == historical[zone_id]
    _assert_allocation_invariants(payload)


def test_forecasting_dataset_unavailable_fallback_when_allowed(client):
    _install_service(records=_make_records(SAMPLE_ZONES), grid_latest=None)
    # Without an origin and without a dataset there is no valid origin at all.
    response = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    assert response.status_code == 503

    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "allow_fallback": True},
    )
    assert response.status_code == 200
    assert response.json()["demand_source"]["type"] == "historical_fallback"


def test_missing_forecast_records_rejected(client):
    _install_service(records=[])
    response = client.post("/fleet/recommendations", json={"horizon_minutes": 60})
    assert response.status_code == 400
    assert "Forecast validation failed" in response.json()["detail"]
    assert "No forecast records" in response.json()["detail"]


def test_missing_forecast_records_fallback_when_allowed(client):
    _install_service(records=[])
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "allow_fallback": True},
    )
    assert response.status_code == 200
    source = response.json()["demand_source"]
    assert source["type"] == "historical_fallback"
    assert "No forecast records" in source["forecast_error"]


def test_wrong_horizon_records_rejected(client):
    """Records produced for another horizon must not be reused."""
    _install_service(records=_make_records(SAMPLE_ZONES, horizon_minutes=360))
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": SAMPLE_ZONES},
    )
    assert response.status_code == 400
    assert "horizon_minutes=60" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("horizon", [30, 90, 120, 0, -60, 1000])
def test_unsupported_horizon_rejected(client, horizon):
    response = client.post("/fleet/recommendations", json={"horizon_minutes": horizon})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unsupported horizon_minutes" in detail
    assert "60" in detail and "180" in detail and "360" in detail


def test_unsupported_horizon_rejected_even_with_fallback(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 90, "allow_fallback": True},
    )
    assert response.status_code == 400


def test_invalid_zone_ids_rejected(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": [61, 999999]},
    )
    assert response.status_code == 400
    assert "Invalid zone_ids" in response.json()["detail"]
    assert "999999" in response.json()["detail"]


def test_invalid_zone_ids_rejected_even_with_fallback(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": [-1], "allow_fallback": True},
    )
    assert response.status_code == 400


def test_duplicate_request_zone_ids_are_deduplicated(client):
    _install_service(records=_make_records([61, 79]))
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": [61, 61, 79, 79]},
    )
    assert response.status_code == 200, response.text
    assert response.json()["demand_source"]["zones_forecasted"] == 2


def test_negative_demand_rejected_by_adapter():
    records = _make_records([61], override={"predicted_rides": -5.0})
    with pytest.raises(ForecastContractError, match="Layer 3 schema"):
        Layer3Adapter.to_fleet_demand(records, 60, [61])


def test_negative_demand_rejected_at_endpoint(client):
    _install_service(records=_make_records(SAMPLE_ZONES, override={"predicted_rides": -5.0}))
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": SAMPLE_ZONES},
    )
    assert response.status_code == 400
    assert "Forecast validation failed" in response.json()["detail"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_demand_rejected(client, value):
    records = _make_records([61], override={"predicted_rides": value})
    with pytest.raises(ForecastContractError):
        Layer3Adapter.to_fleet_demand(records, 60, [61])

    _install_service(records=records)
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": [61]},
    )
    assert response.status_code == 400


def test_finite_guard_catches_infinite_bounds():
    records = _make_records([61], override={"upper_bound": float("inf")})
    with pytest.raises(ForecastContractError, match="upper_bound"):
        Layer3Adapter.to_fleet_demand(records, 60, [61])


def test_duplicate_zone_records_rejected_by_adapter():
    records = _make_records([61]) + _make_records([61])
    with pytest.raises(ForecastContractError, match="Duplicate forecast record"):
        Layer3Adapter.to_fleet_demand(records, 60, [61])


def test_duplicate_zone_records_rejected_at_endpoint(client):
    records = _make_records(SAMPLE_ZONES) + _make_records(SAMPLE_ZONES[:1])
    _install_service(records=records)
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": SAMPLE_ZONES},
    )
    assert response.status_code == 400
    assert "Duplicate forecast record" in response.json()["detail"]


def test_unknown_zone_in_records_rejected():
    records = _make_records([61, 999999])
    with pytest.raises(ForecastContractError, match="outside the fleet zone registry"):
        Layer3Adapter.to_fleet_demand(records, 60, [61])


def test_missing_zone_in_records_rejected():
    records = _make_records([61, 79])
    with pytest.raises(ForecastContractError, match="missing 1 required zone"):
        Layer3Adapter.to_fleet_demand(records, 60, [61, 79, 104])


def test_unexpected_schema_rejected_by_adapter():
    with pytest.raises(ForecastContractError, match="Layer 3 schema"):
        Layer3Adapter.to_fleet_demand([{"zone_id": 61}], 60, [61])


def test_unexpected_schema_rejected_at_endpoint(client):
    _install_service(records=[{"zone_id": 61}])
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "zone_ids": [61]},
    )
    assert response.status_code == 400
    assert "Forecast validation failed" in response.json()["detail"]


def test_inconsistent_target_timestamp_rejected():
    # horizon_minutes claims 60 (so it passes the horizon filter) but the
    # target timestamp claims a 6-hour lead -> contract violation.
    records = _make_records([61], horizon_minutes=60)
    records[0]["target_timestamp"] = "2021-07-01T05:00:00"
    with pytest.raises(ForecastContractError, match="inconsistent with forecast_origin"):
        Layer3Adapter.to_fleet_demand(records, 60, [61])


def test_mixed_forecast_origins_rejected():
    records = _make_records([61], origin="2021-06-30T22:00:00") + _make_records([79])
    with pytest.raises(ForecastContractError, match="different forecast_origin"):
        Layer3Adapter.to_fleet_demand(records, 60, [61, 79])


def test_empty_record_list_rejected():
    with pytest.raises(ForecastContractError, match="No forecast records"):
        Layer3Adapter.to_fleet_demand([], 60, [61])


def test_stale_forecast_origin_rejected_by_adapter():
    records = _make_records([61], origin="2021-06-25T12:00:00")
    with pytest.raises(ForecastContractError, match="Stale forecast"):
        Layer3Adapter.to_fleet_demand(
            records,
            60,
            [61],
            latest_available_timestamp=pd.Timestamp(LATEST_DATA_HOUR),
            max_staleness_hours=24.0,
        )


def test_future_forecast_origin_rejected_by_adapter():
    records = _make_records([61], origin="2021-07-05T12:00:00")
    with pytest.raises(ForecastContractError, match="after the latest available data hour"):
        Layer3Adapter.to_fleet_demand(
            records,
            60,
            [61],
            latest_available_timestamp=pd.Timestamp(LATEST_DATA_HOUR),
            max_staleness_hours=24.0,
        )


def test_stale_forecast_origin_rejected_at_endpoint(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": "2021-01-01T00:00:00"},
    )
    assert response.status_code == 400
    assert "Stale forecast_origin" in response.json()["detail"]


def test_future_forecast_origin_rejected_at_endpoint(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": "2022-01-01T00:00:00"},
    )
    assert response.status_code == 400
    assert "after the newest available data hour" in response.json()["detail"]


def test_recent_forecast_origin_accepted(client):
    response = client.post(
        "/fleet/recommendations",
        json={"horizon_minutes": 60, "forecast_origin": "2021-06-30T20:00:00"},
    )
    assert response.status_code == 200, response.text
    source = response.json()["demand_source"]
    assert source["type"] == "forecast"
    assert source["forecast_origin"] == "2021-06-30T20:00:00"
    assert source["target_timestamp"] == "2021-06-30T21:00:00"


def test_records_from_wrong_origin_rejected():
    records = _make_records([61], origin="2021-06-30T22:00:00")
    with pytest.raises(ForecastContractError, match="instead of the requested"):
        Layer3Adapter.to_fleet_demand(
            records,
            60,
            [61],
            expected_origin=pd.Timestamp("2021-06-30T23:00:00"),
        )


# ---------------------------------------------------------------------------
# 4. Fleet regression around forecast-driven recommendations
# ---------------------------------------------------------------------------


def test_fleet_conservation_with_default_and_forecast_requests(client):
    for body in (None, {"horizon_minutes": 60}, {"horizon_minutes": 180}, {"horizon_minutes": 360}):
        if body is None:
            response = client.post("/fleet/recommendations")
        else:
            response = client.post("/fleet/recommendations", json=body)
        assert response.status_code == 200
        _assert_fleet_conserved()


def test_zero_demand_and_high_demand_scenarios():
    """Allocation stays well-behaved for zero and extreme forecast demand."""
    state = {
        "simulation_id": "test-demand-extremes",
        "fleet_size": 100,
        "zones": [1, 2],
        "idle_cabs": {1: 80, 2: 20},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 80, 2: 20},
        "history": [],
    }

    zero = compute_allocation_recommendations(state, predicted_rides={1: 0.0, 2: 0.0})
    assert len(zero["recommended_movements"]) == 0
    assert zero["unmet_demand_estimate"] == {}
    for zone in zero["zones"]:
        assert zone["shortage_estimate"] == 0

    extreme = compute_allocation_recommendations(state, predicted_rides={1: 10.0, 2: 10000.0})
    assert extreme["unmet_demand_estimate"]["total_unmet_rides"] > 0
    for zone in extreme["zones"]:
        assert zone["shortage_estimate"] >= 0
    _assert_allocation_invariants(extreme)


def test_non_integer_forecast_demand_is_handled():
    """Forecast demand is fractional; cab counts must stay integral."""
    state = {
        "simulation_id": "test-fractional",
        "fleet_size": 40,
        "zones": [1, 2],
        "idle_cabs": {1: 30, 2: 0},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 30, 2: 0},
        "history": [],
    }
    rec = compute_allocation_recommendations(state, predicted_rides={1: 7.5, 2: 25.5})
    assert len(rec["recommended_movements"]) == 1
    move = rec["recommended_movements"][0]
    assert isinstance(move["cab_count"], int)
    assert 1 <= move["cab_count"] <= 5
    _assert_allocation_invariants(rec)


def test_advance_after_forecast_recommendations_still_conserves():
    state = _make_initial_state(fleet_size=500, seed=7)
    demand = _predict_demand_for_interval(state, SIMULATION_INTERVAL_HOURS)
    compute_allocation_recommendations(state, predicted_rides=demand)
    for _ in range(3):
        state = advance_simulation(state)
        total = (
            sum(state["idle_cabs"].values())
            + sum(state["busy_cabs"].values())
            + sum(state["repositioning_cabs"].values())
        )
        assert total == 500


def test_default_recommendations_response_shape_unchanged():
    """The historical path must keep every pre-existing response field."""
    payload = compute_allocation_recommendations(main.fleet_state)
    assert set(payload.keys()) == {
        "simulation_id",
        "prediction_interval",
        "zones",
        "recommended_movements",
        "unmet_demand_estimate",
    }
    zone_keys = set(payload["zones"][0].keys())
    assert zone_keys == {
        "zone_id",
        "predicted_rides",
        "idle_cabs",
        "busy_cabs",
        "repositioning_cabs",
        "estimated_cabs_required",
        "surplus",
        "shortage_estimate",
        "surplus_after_own_demand",
    }
    assert math.isfinite(payload["zones"][0]["predicted_rides"])
