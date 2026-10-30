"""
Ride Demand Forecasting - Fleet Management Prototype

Simulated cab availability and demand-aware fleet allocation.

CRITICAL: This is a SIMULATION prototype. The generated availability does not
represent actual live NYC cab availability. It uses documented assumptions
and historical demand proportions for distribution.

See the project specification (PHASE 1-8) for full design requirements.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration defaults (all configurable via API or environment)
# ---------------------------------------------------------------------------

DEFAULT_FLEET_SIZE = 13_000  # experimental assumption: total Yellow Cab fleet
SIMULATION_START = pd.Timestamp("2021-04-01 00:00:00")
SIMULATION_INTERVAL_HOURS = 1  # prediction interval
RANDOM_SEED = 42  # for reproducibility

# Estimated trip cycle parameters (configurable)
AVG_TRIP_DURATION_MINUTES = 35  # passenger trip avg
AVG_DEADHEAD_MINUTES = 10  # time to next pickup
AVG_REPOSITIONING_MINUTES = 15  # travel between zones

# ---------------------------------------------------------------------------
# Zone configuration from existing project
# ---------------------------------------------------------------------------

ZONES_CSV = BASE_DIR / "taxi_zone_lookup.csv"
HISTORY_PARQUET = BASE_DIR / "ride_demand_weather_hourly.parquet"

# Load zone IDs once at import time
_zones_df = pd.read_csv(ZONES_CSV)
VALID_ZONE_IDS = sorted(_zones_df["LocationID"].tolist())
NUM_ZONES = len(VALID_ZONE_IDS)

# ---------------------------------------------------------------------------
# Historical demand proportions for initial distribution
# ---------------------------------------------------------------------------


def _compute_demand_proportions() -> dict[int, float]:
    """
    Compute zone-level demand proportions from historical data.

    Returns a dict mapping zone_id -> proportion of total rides,
    normalized to sum to 1.0.
    """
    history_path = HISTORY_PARQUET
    if not history_path.exists():
        # Fallback: uniform distribution
        return {zid: 1.0 / NUM_ZONES for zid in VALID_ZONE_IDS}

    # The parquet has ride_count per zone per hour; sum across all hours
    df = pd.read_parquet(history_path)
    zone_rides = df.groupby("PULocationID")["ride_count"].sum()

    # Include all valid zones; missing zones get 0
    proportions: dict[int, float] = {}
    total = 0.0
    for zid in VALID_ZONE_IDS:
        rides = float(zone_rides.get(zid, 0.0))
        proportions[zid] = rides
        total += rides

    if total == 0:
        # Fallback to uniform
        return {zid: 1.0 / NUM_ZONES for zid in VALID_ZONE_IDS}

    # Normalize
    for zid in proportions:
        proportions[zid] /= total

    return proportions


DEMAND_PROPORTIONS = _compute_demand_proportions()

# ---------------------------------------------------------------------------
# Simulation state type & Demand Prediction
# ---------------------------------------------------------------------------

SimulationState = Dict[str, any]

_cached_hourly_avg: Optional[dict[int, float]] = None
DEFAULT_FALLBACK_HOURLY_RIDES_PER_ZONE: float = 5.0  # Documented uniform baseline when parquet missing


def _get_hourly_demand_avg() -> dict[int, float]:
    """Retrieve and cache historical hourly average rides per zone to avoid disk I/O on every step."""
    global _cached_hourly_avg
    if _cached_hourly_avg is not None:
        return _cached_hourly_avg
    if not HISTORY_PARQUET.exists():
        return {}
    df = pd.read_parquet(HISTORY_PARQUET, columns=["PULocationID", "ride_count"])
    _cached_hourly_avg = df.groupby("PULocationID")["ride_count"].mean().to_dict()
    return _cached_hourly_avg


def _predict_demand_for_interval(state: SimulationState, interval_hrs: int = SIMULATION_INTERVAL_HOURS) -> dict[int, int]:
    """
    Predict ride demand for each zone for the given interval.

    Uses historical average rides per hour per zone scaled by interval.
    If historical parquet data is unavailable, falls back to a documented
    uniform baseline rate (DEFAULT_FALLBACK_HOURLY_RIDES_PER_ZONE) across valid zones.
    """
    hourly_avg = _get_hourly_demand_avg()
    predicted: dict[int, int] = {}
    has_history = bool(hourly_avg)

    zones = state.get("zones", VALID_ZONE_IDS)
    for zid in zones:
        if has_history:
            avg = hourly_avg.get(zid, 0.0)
            predicted[zid] = int(round(avg * interval_hrs))
        else:
            predicted[zid] = max(1, int(round(DEFAULT_FALLBACK_HOURLY_RIDES_PER_ZONE * interval_hrs)))

    return predicted


# ---------------------------------------------------------------------------
# Simulation state initialization
# ---------------------------------------------------------------------------


def _make_initial_state(
    fleet_size: int = DEFAULT_FLEET_SIZE,
    seed: int = RANDOM_SEED,
) -> SimulationState:
    """
    Initialize the fleet state across zones using demand proportions.

    Invariants (maintained throughout):
        sum(idle_cabs + busy_cabs + repositioning_cabs across all zones) == fleet_size
    """
    rng = random.Random(seed)

    # Allocate fleet proportionally to historical demand
    proportions = DEMAND_PROPORTIONS

    # Step 1: allocate fleet proportionally
    idle_by_zone: dict[int, int] = {}
    remaining = fleet_size

    # Sort zones by proportion (descending) for deterministic allocation
    sorted_zones = sorted(VALID_ZONE_IDS, key=lambda z: proportions.get(z, 0), reverse=True)

    for i, zid in enumerate(sorted_zones):
        if i < len(sorted_zones) - 1:
            # Allocate to all but the last zone
            prop = proportions.get(zid, 1.0 / NUM_ZONES)
            allocated = int(round(prop * fleet_size))
            # Leave at least 1 for each remaining zone
            allocated = min(allocated, remaining - (len(sorted_zones) - i - 1))
        else:
            # Last zone gets everything remaining
            allocated = remaining

        idle_by_zone[zid] = max(0, allocated)
        remaining -= idle_by_zone[zid]

    # Step 2: add small random redistribution between zones
    # Move a few cabs from higher-proportion zones to lower-proportion zones
    num_redistribution_moves = max(1, int(0.1 * fleet_size // NUM_ZONES))

    for _ in range(num_redistribution_moves):
        # Pick source zone with at least 2 cabs
        source_candidates = [z for z in VALID_ZONE_IDS if idle_by_zone.get(z, 0) >= 2]
        if not source_candidates:
            break
        source = rng.choice(source_candidates)
        # Pick destination zone that is not the source
        dest_candidates = [z for z in VALID_ZONE_IDS if z != source]
        dest = rng.choice(dest_candidates)
        # Move 1-2 cabs
        move_amount = rng.randint(1, min(2, idle_by_zone[source] - 1))

        idle_by_zone[source] -= move_amount
        idle_by_zone[dest] = idle_by_zone.get(dest, 0) + move_amount

    # Step 3: ensure no negative values and exact total
    for zid in VALID_ZONE_IDS:
        if idle_by_zone.get(zid, 0) < 0:
            idle_by_zone[zid] = 0

    # Redistribute any remaining fleet to ensure exact total
    current_total = sum(idle_by_zone.get(z, 0) for z in VALID_ZONE_IDS)
    diff = fleet_size - current_total

    if diff != 0:
        # Absorb diff into the zone with highest proportion
        target = max(proportions, key=proportions.get)
        idle_by_zone[target] += diff
        if idle_by_zone[target] < 0:
            idle_by_zone[target] = 0

    # Verify final total
    assert sum(idle_by_zone.get(z, 0) for z in VALID_ZONE_IDS) == fleet_size, (
        f"Fleet size mismatch after allocation: {sum(idle_by_zone.values())} != {fleet_size}"
    )

    # All cabs start as idle; no cabs are initially busy or repositioning
    busy_by_zone: dict[int, int] = {zid: 0 for zid in VALID_ZONE_IDS}
    repositioning_by_zone: dict[int, int] = {zid: 0 for zid in VALID_ZONE_IDS}
    total_by_zone: dict[int, int] = {
        zid: idle_by_zone.get(zid, 0) + busy_by_zone.get(zid, 0) + repositioning_by_zone.get(zid, 0)
        for zid in VALID_ZONE_IDS
    }

    # Double-check invariant
    assert sum(total_by_zone.values()) == fleet_size, (
        f"Total invariant violated: {sum(total_by_zone.values())} != {fleet_size}"
    )

    state: SimulationState = {
        "simulation_id": f"sim-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "fleet_size": fleet_size,
        "timestamp": datetime.utcnow(),
        "zones": VALID_ZONE_IDS,
        "idle_cabs": idle_by_zone,
        "busy_cabs": busy_by_zone,
        "repositioning_cabs": repositioning_by_zone,
        "total_cabs_by_zone": total_by_zone,
        "history": [],
        "trip_schedule": [],  # pending trip completions
        "reposition_schedule": [],  # pending repositioning completions
        "predicted_rides": _predict_demand_for_interval({"zones": VALID_ZONE_IDS}, SIMULATION_INTERVAL_HOURS),
    }

    # Record initial snapshot
    state["history"].append(_make_snapshot(state))

    return state


def _make_snapshot(state: SimulationState) -> Dict[str, any]:
    """Create a serializable snapshot of the current fleet state."""
    return {
        "timestamp": state["timestamp"].isoformat(),
        "simulation_id": state["simulation_id"],
        "fleet_size": state["fleet_size"],
        "zones": state["zones"],
        "idle_cabs": {z: state["idle_cabs"].get(z, 0) for z in state["zones"]},
        "busy_cabs": {z: state["busy_cabs"].get(z, 0) for z in state["zones"]},
        "repositioning_cabs": {
            z: state["repositioning_cabs"].get(z, 0) for z in state["zones"]
        },
        "total_cabs_by_zone": state["total_cabs_by_zone"],
    }


def _snapshot_for_export(state: SimulationState) -> List[Dict[str, any]]:
    """Export all zone snapshots from the history."""
    exports: List[Dict[str, any]] = []
    for snap in state["history"]:
        for z in state["zones"]:
            exports.append(
                {
                    "timestamp": snap["timestamp"],
                    "simulation_id": snap["simulation_id"],
                    "zone_id": z,
                    "idle_cabs": snap["idle_cabs"].get(z, 0),
                    "busy_cabs": snap["busy_cabs"].get(z, 0),
                    "repositioning_cabs": snap["repositioning_cabs"].get(z, 0),
                    "total_cabs": snap["total_cabs_by_zone"].get(z, 0),
                }
            )
    return exports

# ---------------------------------------------------------------------------
# Trip and repositioning modeling parameters
# ---------------------------------------------------------------------------


def _trip_cycle_duration_minutes() -> int:
    """Estimate total service cycle duration in minutes."""
    return int(AVG_TRIP_DURATION_MINUTES + AVG_DEADHEAD_MINUTES)


def _repositioning_duration_minutes() -> int:
    """Estimate repositioning travel time between zones in minutes."""
    return int(AVG_REPOSITIONING_MINUTES)


# ---------------------------------------------------------------------------
# Simulation advancement
# ---------------------------------------------------------------------------


def advance_simulation(state: SimulationState) -> SimulationState:
    """
    Advance the simulation by one interval.

    Models:
    - Repositioning completions: repositioning cabs arrive and become idle in their destination zone
    - Trip completions: busy cabs finish trips and return to idle
      (handled via trip_schedule if present, or released after the 45-minute service cycle)
    - New trips begin: idle cabs become busy based on predicted demand
    - New repositioning moves: idle cabs reposition towards predicted demand

    Invariants (strictly maintained):
        sum(idle_cabs + busy_cabs + repositioning_cabs) == fleet_size
        total_cabs_by_zone[z] == idle[z] + busy[z] + repositioning[z]
    """
    prediction_interval_hrs = SIMULATION_INTERVAL_HOURS

    # --- Step 1: Handle completions from the previous interval ---
    # Cabs that were repositioning arrive and become idle in their destination zone
    completing_reposition: dict[int, int] = {}
    for zid in state["zones"]:
        completing_reposition[zid] = state["repositioning_cabs"].get(zid, 0)

    # Cabs that were busy complete their trips.
    # We inspect trip_schedule: if active scheduled trips exist, we check if
    # their duration/schedule indicates trips are still in progress or completed.
    trip_schedule = state.get("trip_schedule", [])
    new_trip_schedule = []
    completing_busy: dict[int, int] = {zid: 0 for zid in state["zones"]}
    remaining_busy: dict[int, int] = {zid: 0 for zid in state["zones"]}

    # Track which busy cabs in each zone are tied to scheduled multi-interval trips
    scheduled_cabs_by_zone: dict[int, int] = {zid: 0 for zid in state["zones"]}

    if trip_schedule:
        for trip in trip_schedule:
            cabs = trip.get("cabs", 1)
            dest = trip.get("destination_zone", trip.get("zone_id", trip.get("dropoff_zone", state["zones"][0])))
            curr_loc = trip.get("current_zone", trip.get("origin_zone", dest))
            if curr_loc not in state["zones"]:
                curr_loc = state["zones"][0]
            if dest not in state["zones"]:
                dest = state["zones"][0]

            # Mark these cabs as scheduled in their current zone
            scheduled_cabs_by_zone[curr_loc] = scheduled_cabs_by_zone.get(curr_loc, 0) + cabs

            rem = trip.get("intervals_remaining", 1) - 1
            if rem <= 0:
                # Trip completed: cabs become idle in destination zone
                completing_busy[dest] = completing_busy.get(dest, 0) + cabs
            else:
                # Trip still in progress: keep in schedule and in current zone busy_cabs
                updated_trip = dict(trip)
                updated_trip["intervals_remaining"] = rem
                updated_trip["current_zone"] = curr_loc
                new_trip_schedule.append(updated_trip)
                remaining_busy[curr_loc] = remaining_busy.get(curr_loc, 0) + cabs

    # Standard cycle completions:
    # Service cycle is 45 minutes (35 min trip + 10 min deadhead), which is less than the 60-minute interval.
    # Any busy cabs not held in active trip_schedule complete their cycle and return to idle.
    for zid in state["zones"]:
        busy_in_zone = state["busy_cabs"].get(zid, 0)
        unaccounted_busy = max(0, busy_in_zone - scheduled_cabs_by_zone.get(zid, 0))
        completing_busy[zid] = completing_busy.get(zid, 0) + unaccounted_busy

    # New idle counts include previously idle cabs, arriving repositioning cabs, and completed trips
    new_idle_after_reposition: dict[int, int] = {}
    for zid in state["zones"]:
        new_idle_after_reposition[zid] = (
            state["idle_cabs"].get(zid, 0)
            + completing_reposition.get(zid, 0)
            + completing_busy.get(zid, 0)
        )

    # --- Step 2: Predict demand and assign idle cabs as busy ---
    predicted_rides_by_zone: dict[int, int] = _predict_demand_for_interval(
        state, prediction_interval_hrs
    )

    # Capacity per cab per interval
    capacity_per_cab = _capacity_per_cab_per_interval()

    # Assign cabs to predicted rides
    new_busy_after_demand: dict[int, int] = {}
    new_idle_after_demand: dict[int, int] = {}

    for zid in state["zones"]:
        rides = predicted_rides_by_zone.get(zid, 0)
        idle = new_idle_after_reposition.get(zid, 0)

        # Estimated cabs needed to serve predicted rides
        cabs_needed = max(1, int(np.ceil(rides / capacity_per_cab))) if rides > 0 else 0

        # Don't assign more than available idle cabs
        cabs_actually_assigned = min(cabs_needed, idle)

        # Total busy in this zone = any in-progress trips + newly assigned trips
        new_busy_after_demand[zid] = remaining_busy.get(zid, 0) + cabs_actually_assigned
        new_idle_after_demand[zid] = idle - cabs_actually_assigned

    # --- Step 3: Repositioning moves ---
    # A percentage of the fleet repositions between zones based on demand
    reposition_amount = max(1, int(0.03 * state["fleet_size"] // NUM_ZONES))  # 3% per zone

    # Reset repositioning counts; we'll rebuild them
    new_repositioning: dict[int, int] = {zid: 0 for zid in state["zones"]}

    # Move idle cabs to repositioning, destination weighted by demand proportions
    for zid in state["zones"]:
        current_idle = new_idle_after_demand.get(zid, 0)
        move_out = min(reposition_amount, current_idle)
        new_idle_after_demand[zid] = current_idle - move_out

        for _ in range(move_out):
            dest_zid = rng_choice_weighted(DEMAND_PROPORTIONS)
            new_repositioning[dest_zid] = new_repositioning.get(dest_zid, 0) + 1

    # --- Step 4: Recompute totals and enforce invariants ---
    new_total: dict[int, int] = {}
    for zid in state["zones"]:
        new_total[zid] = (
            new_idle_after_demand.get(zid, 0)
            + new_busy_after_demand.get(zid, 0)
            + new_repositioning.get(zid, 0)
        )

    # Invariant check: total fleet must be conserved
    fleet_total = sum(new_total.values())
    assert fleet_total == state["fleet_size"], (
        f"Fleet invariant violated after advance: {fleet_total} != {state['fleet_size']}"
    )

    # Build new state
    new_state: SimulationState = {
        "simulation_id": state["simulation_id"],
        "fleet_size": state["fleet_size"],
        "timestamp": datetime.utcnow(),
        "zones": state["zones"],
        "idle_cabs": new_idle_after_demand,
        "busy_cabs": new_busy_after_demand,
        "repositioning_cabs": new_repositioning,
        "total_cabs_by_zone": new_total,
        "history": state["history"],
        "trip_schedule": new_trip_schedule,
        "reposition_schedule": state.get("reposition_schedule", []),
        "predicted_rides": predicted_rides_by_zone,
    }

    # Record snapshot
    new_state["history"].append(_make_snapshot(new_state))

    return new_state


def _capacity_per_cab_per_interval() -> float:
    """
    Estimate how many rides one cab can handle in the prediction interval.

    First approximation: interval_duration / expected_service_cycle_duration.

    Service cycle = trip duration + deadhead time.
    """
    interval_hrs = SIMULATION_INTERVAL_HOURS
    cycle_mins = _trip_cycle_duration_minutes()
    cycle_hrs = cycle_mins / 60.0

    if cycle_hrs >= interval_hrs:
        return 1.0  # one cab can do at most one full cycle per interval
    else:
        return interval_hrs / cycle_hrs  # e.g., 2.5 rides per cab


# ---------------------------------------------------------------------------
# Zone selection helper
# ---------------------------------------------------------------------------


def rng_choice_weighted(proportions: dict[int, float]) -> int:
    """Pick a zone based on demand proportions."""
    zones_list = list(proportions.keys())
    weights = [proportions[z] for z in zones_list]
    if sum(weights) == 0:
        return random.randint(0, len(zones_list) - 1)
    return random.choices(zones_list, weights=weights, k=1)[0]


def rng_choice_weighted_uniform(n: int) -> int:
    """Pick uniformly from n zones."""
    return random.randint(0, n - 1)

# ---------------------------------------------------------------------------
# Allocation recommendations
# ---------------------------------------------------------------------------


def compute_allocation_recommendations(
    state: SimulationState,
    predicted_rides: Optional[Dict[int, float]] = None,
) -> Dict[str, any]:
    """
    Compute deterministic cab allocation recommendations.

    Identifies zones with predicted shortages and surplus idle cabs,
    and recommends movements from donor to shortage zones.

    Demand resolution order:
      1. `predicted_rides` argument, when supplied (forecast-driven callers).
      2. `state["predicted_rides"]`, when present (historical hourly averages).
      3. `_predict_demand_for_interval()` recomputed on the fly.

    Supplying `predicted_rides` only overrides the demand input. The allocation
    mathematics (fleet conservation, donor selection, shortage/surplus rules,
    movement cap) is identical for forecast and historical demand.

    Returns a dict with:
    - per-zone metrics (predicted_rides, idle_cabs, estimated_cabs_required, surplus, shortage_estimate)
    - recommended_movements (from_zone, to_zone, cab_count, reason)
    - unmet_demand_estimate (total unmet rides, affected zones)
    """
    prediction_interval_hrs = SIMULATION_INTERVAL_HOURS
    if predicted_rides is not None:
        # Caller-supplied demand (validated upstream). Never mutated here.
        zone_demand = predicted_rides
    elif "predicted_rides" in state and state["predicted_rides"] is not None:
        zone_demand = state["predicted_rides"]
    else:
        zone_demand = _predict_demand_for_interval(
            state, prediction_interval_hrs
        )

    # Compute per-zone metrics
    zones = state["zones"]
    zone_metrics: List[Dict[str, any]] = []
    capacity_per_cab = _capacity_per_cab_per_interval()

    for zid in zones:
        predicted = zone_demand.get(zid, 0)
        idle = state["idle_cabs"].get(zid, 0)
        busy = state["busy_cabs"].get(zid, 0)
        reposition = state["repositioning_cabs"].get(zid, 0)
        total = state["total_cabs_by_zone"].get(zid, 0)

        # Estimate cabs required to serve predicted rides
        if predicted > 0:
            cabs_required = max(1, int(np.ceil(predicted / capacity_per_cab)))
        else:
            cabs_required = 0

        # Busy cabs are already serving demand in this zone for the current interval
        needed_from_idle = max(0, cabs_required - busy)
        surplus = max(0, idle - needed_from_idle)
        # Unmet demand: rides that won't be served given current idle + busy capacity
        total_capacity = (idle + busy) * capacity_per_cab
        unmet = max(0.0, float(predicted - total_capacity)) if predicted > 0 else 0.0

        zone_metrics.append(
            {
                "zone_id": zid,
                "predicted_rides": predicted,
                "idle_cabs": idle,
                "busy_cabs": busy,
                "repositioning_cabs": reposition,
                "estimated_cabs_required": cabs_required,
                "surplus": surplus,
                "shortage_estimate": unmet,  # rides that won't be served
                "surplus_after_own_demand": surplus,
            }
        )

    # Identify shortage and surplus zones
    shortage_zones: List[Dict[str, any]] = []
    surplus_zones: List[Dict[str, any]] = []

    for m in zone_metrics:
        # A zone is classified as a shortage zone IF AND ONLY IF its estimated unmet demand is greater than zero
        is_shortage = m["shortage_estimate"] > 0
        # Surplus: idle cabs remaining after covering own predicted demand
        has_surplus = m["surplus_after_own_demand"] > 0

        if is_shortage:
            shortage_zones.append(m)
        if has_surplus:
            surplus_zones.append(m)

    # Greedy allocation: move cabs from surplus to shortage zones
    recommendations: List[Dict[str, any]] = []
    assigned_surplus: dict[int, int] = {m["zone_id"]: 0 for m in surplus_zones}

    # Sort shortage zones by severity (highest unmet demand first)
    shortage_sorted = sorted(shortage_zones, key=lambda m: m["shortage_estimate"], reverse=True)

    for shortage in shortage_sorted:
        z_to = shortage["zone_id"]

        # Convert unmet rides to the number of cabs required using the capacity-per-cab estimate
        unmet_rides = shortage["shortage_estimate"]
        if unmet_rides <= 0:
            continue
        cabs_needed_for_shortage = max(1, int(np.ceil(unmet_rides / capacity_per_cab)))

        # Find the best donor zone: has surplus after own demand, and is not the destination
        z_from = None
        available_from = 0
        for surplus in surplus_zones:
            s_zid = surplus["zone_id"]
            if s_zid == z_to:
                continue  # cannot donate to itself
            remaining_donor_surplus = surplus["surplus_after_own_demand"] - assigned_surplus.get(s_zid, 0)
            if remaining_donor_surplus > 0:
                z_from = s_zid
                available_from = remaining_donor_surplus
                break

        if z_from is None or available_from <= 0:
            # No eligible donor available
            continue

        # Cap movement at 5 for prototype to avoid excessive shuffling, while respecting available donor surplus
        cab_count = min(cabs_needed_for_shortage, available_from, 5)

        if cab_count >= 1:
            assigned_surplus[z_from] = assigned_surplus.get(z_from, 0) + cab_count
            recommendations.append(
                {
                    "from_zone": z_from,
                    "to_zone": z_to,
                    "cab_count": cab_count,
                    "reason": f"Estimated shortage of {shortage['shortage_estimate']} rides in zone {z_to}",
                }
            )

    # Compute unmet demand aggregate
    total_unmet = sum(m["shortage_estimate"] for m in zone_metrics)
    remaining_shortage: Dict[str, any] = {}
    if total_unmet > 0:
        remaining_shortage = {
            "total_unmet_rides": total_unmet,
            "affected_zones": [
                m["zone_id"] for m in zone_metrics if m["shortage_estimate"] > 0
            ],
        }

    return {
        "simulation_id": state["simulation_id"],
        "prediction_interval": f"{SIMULATION_INTERVAL_HOURS} hour",
        "zones": zone_metrics,
        "recommended_movements": recommendations,
        "unmet_demand_estimate": remaining_shortage,
    }
