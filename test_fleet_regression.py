"""
Regression tests for Fleet Management Prototype.

Validates:
1. Fix 1: Fleet conservation at init and across >= 3 consecutive advances,
   handling of trip schedules, and zone vs global count agreement.
2. Fix 2: Allocation recommendations (no phantom moves in no-shortage scenarios,
   conversion of unmet rides to cabs using capacity, donor surplus bounds,
   zero demand, multiple shortages, insufficient donor supply, immutability).
3. Fix 3: Demand estimation consistency between init and advance, state key
   preservation, and missing-data fallback behavior.
4. API response compatibility for all fleet endpoints.
"""

import copy
import numpy as np
import pytest
from fastapi.testclient import TestClient

import fleet_simulation
from fleet_simulation import (
    _make_initial_state,
    advance_simulation,
    compute_allocation_recommendations,
    _capacity_per_cab_per_interval,
    _predict_demand_for_interval,
    _trip_cycle_duration_minutes,
    DEFAULT_FLEET_SIZE,
    SIMULATION_INTERVAL_HOURS,
    VALID_ZONE_IDS,
)
from main import app


# ---------------------------------------------------------------------------
# FIX 1: Fleet Conservation & Cab Lifecycle Tests
# ---------------------------------------------------------------------------


def test_fleet_conservation_init():
    """Verify fleet conservation and per-zone totals at initialization."""
    state = _make_initial_state(fleet_size=13000, seed=42)
    assert state["fleet_size"] == 13000

    total_idle = sum(state["idle_cabs"].values())
    total_busy = sum(state["busy_cabs"].values())
    total_repo = sum(state["repositioning_cabs"].values())

    assert total_idle + total_busy + total_repo == 13000
    assert sum(state["total_cabs_by_zone"].values()) == 13000

    # Check each individual zone
    for z in state["zones"]:
        zone_sum = (
            state["idle_cabs"].get(z, 0)
            + state["busy_cabs"].get(z, 0)
            + state["repositioning_cabs"].get(z, 0)
        )
        assert state["total_cabs_by_zone"][z] == zone_sum
        assert state["idle_cabs"][z] >= 0
        assert state["busy_cabs"][z] >= 0
        assert state["repositioning_cabs"][z] >= 0


def test_fleet_conservation_consecutive_advances():
    """Verify fleet conservation across at least three consecutive simulation advances."""
    state = _make_initial_state(fleet_size=13000, seed=42)

    for step in range(1, 5):  # 4 consecutive advances
        state = advance_simulation(state)

        total_idle = sum(state["idle_cabs"].values())
        total_busy = sum(state["busy_cabs"].values())
        total_repo = sum(state["repositioning_cabs"].values())

        assert (
            total_idle + total_busy + total_repo == 13000
        ), f"Advance {step} failed: sum={total_idle+total_busy+total_repo} != 13000"
        assert (
            sum(state["total_cabs_by_zone"].values()) == 13000
        ), f"Advance {step} zone totals mismatch"

        # Verify per-zone agreement and non-negativity
        for z in state["zones"]:
            zone_sum = (
                state["idle_cabs"].get(z, 0)
                + state["busy_cabs"].get(z, 0)
                + state["repositioning_cabs"].get(z, 0)
            )
            assert state["total_cabs_by_zone"][z] == zone_sum
            assert state["idle_cabs"][z] >= 0
            assert state["busy_cabs"][z] >= 0
            assert state["repositioning_cabs"][z] >= 0


def test_trip_schedule_handling_in_progress():
    """Verify that trips with remaining intervals in trip_schedule are not prematurely moved to idle."""
    state = _make_initial_state(fleet_size=100, seed=42)

    # Pick a zone that has sufficient idle cabs to schedule a trip
    target_zone = max(state["idle_cabs"], key=state["idle_cabs"].get)
    dest_zone = [z for z in state["zones"] if z != target_zone][0]
    cabs_to_trip = min(5, state["idle_cabs"][target_zone])

    state["idle_cabs"][target_zone] -= cabs_to_trip
    state["busy_cabs"][target_zone] = cabs_to_trip
    state["total_cabs_by_zone"][target_zone] = (
        state["idle_cabs"][target_zone] + state["busy_cabs"][target_zone]
    )
    state["trip_schedule"] = [
        {
            "cabs": cabs_to_trip,
            "origin_zone": target_zone,
            "destination_zone": dest_zone,
            "intervals_remaining": 2,  # Should remain busy on next advance
        }
    ]

    # Advance 1: trip should still be in progress (intervals_remaining decremented to 1)
    state1 = advance_simulation(state)
    assert any(
        t.get("intervals_remaining") == 1 for t in state1["trip_schedule"]
    ), "Trip schedule should retain in-progress trip with intervals_remaining=1"
    assert sum(state1["total_cabs_by_zone"].values()) == 100

    # Advance 2: trip should complete and arrive at dest_zone
    state2 = advance_simulation(state1)
    assert len(state2["trip_schedule"]) == 0, "Completed trip should be removed from schedule"
    assert sum(state2["total_cabs_by_zone"].values()) == 100


# ---------------------------------------------------------------------------
# FIX 2: Allocation Recommendations Tests
# ---------------------------------------------------------------------------


def _mock_recs(state, mock_demand):
    orig = fleet_simulation._predict_demand_for_interval
    try:
        fleet_simulation._predict_demand_for_interval = lambda s, h: mock_demand
        return compute_allocation_recommendations(state)
    finally:
        fleet_simulation._predict_demand_for_interval = orig


def test_recs_zero_demand():
    """Zero demand in all zones produces zero movements and zero unmet demand."""
    state = {
        "simulation_id": "test-zero",
        "fleet_size": 100,
        "zones": [1, 2],
        "idle_cabs": {1: 50, 2: 50},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 50, 2: 50},
        "history": [],
    }
    rec = _mock_recs(state, {1: 0, 2: 0})
    assert len(rec["recommended_movements"]) == 0
    assert rec["unmet_demand_estimate"] == {}
    for m in rec["zones"]:
        assert m["shortage_estimate"] == 0
        assert m["surplus"] == 50


def test_recs_no_shortage_no_phantom_movements():
    """When idle capacity comfortably exceeds demand, no phantom movements are generated."""
    state = {
        "simulation_id": "test-no-shortage",
        "fleet_size": 200,
        "zones": [10, 20],
        "idle_cabs": {10: 100, 20: 100},
        "busy_cabs": {10: 0, 20: 0},
        "repositioning_cabs": {10: 0, 20: 0},
        "total_cabs_by_zone": {10: 100, 20: 100},
        "history": [],
    }
    rec = _mock_recs(state, {10: 10, 20: 10})
    # Both zones have 100 idle cabs and demand of 10. No movements should occur.
    assert len(rec["recommended_movements"]) == 0
    assert rec["unmet_demand_estimate"] == {}


def test_recs_unmet_rides_to_cabs_conversion():
    """Unmet rides are converted to required cabs using capacity_per_cab (ceil(unmet / cap))."""
    cap = _capacity_per_cab_per_interval()  # 1.3333333333333333
    state = {
        "simulation_id": "test-conv",
        "fleet_size": 100,
        "zones": [1, 2],
        "idle_cabs": {1: 80, 2: 0},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 80, 2: 0},
        "history": [],
    }
    # Zone 1: demand=0, surplus = 80
    # Zone 2: demand=4 rides -> unmet = 4 rides. Cabs needed = ceil(4 / 1.3333) = ceil(3) = 3 cabs.
    rec = _mock_recs(state, {1: 0, 2: 4})
    assert len(rec["recommended_movements"]) == 1
    mov = rec["recommended_movements"][0]
    assert mov["from_zone"] == 1
    assert mov["to_zone"] == 2
    assert mov["cab_count"] == 3  # exactly 3 cabs, NOT 4 cabs!


def test_recs_one_donor_one_shortage():
    """One donor with surplus and one shortage zone with cap at 5."""
    state = {
        "simulation_id": "test-one-one",
        "fleet_size": 100,
        "zones": [1, 2],
        "idle_cabs": {1: 80, 2: 10},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 80, 2: 10},
        "history": [],
    }
    # Zone 1: demand=10 (needs 8 cabs, surplus = 72)
    # Zone 2: demand=40 (needs 30 cabs, idle=10, unmet = 26.67 rides, needed = 20 cabs)
    rec = _mock_recs(state, {1: 10, 2: 40})
    assert len(rec["recommended_movements"]) == 1
    mov = rec["recommended_movements"][0]
    assert mov["from_zone"] == 1
    assert mov["to_zone"] == 2
    assert mov["cab_count"] == 5  # capped at prototype limit of 5


def test_recs_insufficient_donor_supply():
    """Donor does not allocate more cabs than its available surplus."""
    state = {
        "simulation_id": "test-insuff",
        "fleet_size": 20,
        "zones": [1, 2],
        "idle_cabs": {1: 10, 2: 0},
        "busy_cabs": {1: 0, 2: 0},
        "repositioning_cabs": {1: 0, 2: 0},
        "total_cabs_by_zone": {1: 10, 2: 0},
        "history": [],
    }
    # Zone 1: demand=10 -> needs 8 cabs. Surplus = 10 - 8 = 2 cabs.
    # Zone 2: demand=30 -> needs 23 cabs.
    rec = _mock_recs(state, {1: 10, 2: 30})
    assert len(rec["recommended_movements"]) == 1
    mov = rec["recommended_movements"][0]
    assert mov["from_zone"] == 1
    assert mov["to_zone"] == 2
    assert mov["cab_count"] == 2  # exactly donor's surplus (2), never exceeds 2!


def test_recs_state_immutability():
    """Recommendations generation does not mutate the input simulation state."""
    state = _make_initial_state(fleet_size=1000, seed=42)
    state_before = copy.deepcopy(state)

    rec1 = compute_allocation_recommendations(state)
    rec2 = compute_allocation_recommendations(state)

    assert state == state_before, "Simulation state was mutated by compute_allocation_recommendations"
    assert rec1 == rec2, "Repeated calls with unchanged state returned different recommendations"


# ---------------------------------------------------------------------------
# FIX 3: Demand Estimation Consistency Tests
# ---------------------------------------------------------------------------


def test_demand_consistency_init_and_advance():
    """Predicted demand is consistent between initialization and advancement."""
    state = _make_initial_state(fleet_size=13000, seed=42)
    assert "predicted_rides" in state
    init_predicted = state["predicted_rides"]

    # Advance the simulation
    state_advanced = advance_simulation(state)
    assert "predicted_rides" in state_advanced
    adv_predicted = state_advanced["predicted_rides"]

    # Both should use the exact same zone-level prediction values
    for z in state["zones"]:
        assert init_predicted[z] == adv_predicted[z], (
            f"Demand inconsistency in zone {z}: init={init_predicted[z]} vs adv={adv_predicted[z]}"
        )


def test_missing_data_fallback():
    """Missing parquet produces documented uniform baseline demand without collapsing to 0."""
    orig_path = fleet_simulation.HISTORY_PARQUET
    orig_cache = fleet_simulation._cached_hourly_avg
    try:
        from pathlib import Path
        fleet_simulation.HISTORY_PARQUET = Path("non_existent_file.parquet")
        fleet_simulation._cached_hourly_avg = None

        state = {"zones": [1, 2, 3]}
        pred = _predict_demand_for_interval(state, 1)

        assert len(pred) == 3
        for z in [1, 2, 3]:
            # Should produce documented uniform baseline (5 rides/hr)
            assert pred[z] == 5
    finally:
        fleet_simulation.HISTORY_PARQUET = orig_path
        fleet_simulation._cached_hourly_avg = orig_cache


# ---------------------------------------------------------------------------
# API Endpoints Compatibility Tests
# ---------------------------------------------------------------------------


def test_api_endpoints_workflow():
    """Verify existing fleet API endpoints respond with expected schemas and status codes."""
    client = TestClient(app)

    # 1. POST /fleet/init
    r_init = client.post("/fleet/init")
    assert r_init.status_code == 200
    data_init = r_init.json()
    assert data_init["status"] == "initialized"
    assert data_init["fleet_size"] == 13000
    assert "simulation_id" in data_init

    # 2. GET /fleet/current
    r_curr = client.get("/fleet/current")
    assert r_curr.status_code == 200
    data_curr = r_curr.json()
    assert data_curr["simulation_id"] == data_init["simulation_id"]
    assert data_curr["fleet_size"] == 13000
    assert "predicted_rides" in data_curr
    assert "availability" in data_curr
    assert data_curr["fleet_totals"]["conservation_check"] is True

    # 3. POST /fleet/recommendations
    r_rec = client.post("/fleet/recommendations")
    assert r_rec.status_code == 200
    data_rec = r_rec.json()
    assert "zones" in data_rec
    assert "recommended_movements" in data_rec
    assert "unmet_demand_estimate" in data_rec

    # 4. POST /fleet/advance (Call three consecutive times to ensure multi-step stability)
    for step in range(1, 4):
        r_adv = client.post("/fleet/advance")
        assert r_adv.status_code == 200, f"Advance step {step} failed: {r_adv.text}"
        data_adv = r_adv.json()
        assert data_adv["status"] == "advanced"
        assert data_adv["conservation_check"] is True

    # Verify current state after advances is still 100% conserved
    r_curr_after = client.get("/fleet/current")
    assert r_curr_after.status_code == 200
    assert r_curr_after.json()["fleet_totals"]["conservation_check"] is True
    assert (
        r_curr_after.json()["fleet_totals"]["total_idle"]
        + r_curr_after.json()["fleet_totals"]["total_busy"]
        + r_curr_after.json()["fleet_totals"]["total_repositioning"]
        == 13000
    )

    # 5. GET /fleet/export
    r_exp = client.get("/fleet/export")
    assert r_exp.status_code == 200
    data_exp = r_exp.json()
    assert "snapshots" in data_exp
    assert data_exp["snapshot_count"] >= 4

    # 6. POST /fleet/reset
    r_reset = client.post("/fleet/reset")
    assert r_reset.status_code == 200
    assert r_reset.json()["status"] == "reset"


def test_configurable_fleet_size_initialization():
    """Verify that fleet initialization accepts configurable fleet sizes (e.g. 16k, 20k, 25k)."""
    client = TestClient(app)
    for target_size in [16000, 20000, 25000]:
        r = client.post("/fleet/init", json={"fleet_size": target_size})
        assert r.status_code == 200
        data = r.json()
        assert data["fleet_size"] == target_size

        r_curr = client.get("/fleet/current")
        assert r_curr.status_code == 200
        data_curr = r_curr.json()
        assert data_curr["fleet_size"] == target_size
        assert data_curr["fleet_totals"]["total_fleet"] == target_size
        assert data_curr["fleet_totals"]["conservation_check"] is True


def test_recs_count_and_validity_at_20k():
    """Verify that at 20k fleet size, genuine recommendations (~10-20) are generated and adhere to all validity invariants."""
    client = TestClient(app)
    r_init = client.post("/fleet/init", json={"fleet_size": 20000})
    assert r_init.status_code == 200

    r_recs = client.post("/fleet/recommendations")
    assert r_recs.status_code == 200
    data = r_recs.json()

    movements = data["recommended_movements"]
    zones = {z["zone_id"]: z for z in data["zones"]}

    # Should produce a meaningful number of recommendations (~10-20)
    assert 10 <= len(movements) <= 25, f"Expected 10-25 recommendations, got {len(movements)}"

    donor_donated = {}
    for m in movements:
        from_z = m["from_zone"]
        to_z = m["to_zone"]
        cabs = m["cab_count"]

        # 1. No self-movements
        assert from_z != to_z, f"Self-movement detected: {from_z} -> {to_z}"

        # 2. Positive cab count respecting movement cap of 5
        assert 1 <= cabs <= 5, f"Invalid cab count: {cabs}"

        # 3. Donor had surplus
        assert zones[from_z]["surplus_after_own_demand"] > 0

        # 4. Recipient had actual shortage
        assert zones[to_z]["shortage_estimate"] > 0

        donor_donated[from_z] = donor_donated.get(from_z, 0) + cabs

    # 5. Donor never donates more than its available surplus
    for donor_id, total_donated in donor_donated.items():
        assert total_donated <= zones[donor_id]["surplus_after_own_demand"], (
            f"Donor {donor_id} over-donated: {total_donated} > {zones[donor_id]['surplus_after_own_demand']}"
        )


def test_conservation_four_consecutive_advances():
    """Verify fleet conservation and non-empty valid recommendations over 4 consecutive advances."""
    client = TestClient(app)
    fleet_size = 20000
    client.post("/fleet/init", json={"fleet_size": fleet_size})

    for adv in range(1, 5):
        r_adv = client.post("/fleet/advance")
        assert r_adv.status_code == 200
        assert r_adv.json()["conservation_check"] is True

        r_curr = client.get("/fleet/current")
        assert r_curr.status_code == 200
        ft = r_curr.json()["fleet_totals"]
        assert ft["conservation_check"] is True
        assert ft["total_idle"] + ft["total_busy"] + ft["total_repositioning"] == fleet_size

        r_rec = client.post("/fleet/recommendations")
        assert r_rec.status_code == 200
        rec_data = r_rec.json()
        assert len(rec_data["recommended_movements"]) >= 10, (
            f"Expected at least 10 recommendations after advance {adv}, got {len(rec_data['recommended_movements'])}"
        )
