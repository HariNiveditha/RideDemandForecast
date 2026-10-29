import sys
sys.path.insert(0, r'.')
import numpy as np
from fleet_simulation import (
    _make_initial_state,
    advance_simulation,
    compute_allocation_recommendations,
    _capacity_per_cab_per_interval,
    _predict_demand_for_interval,
    _trip_cycle_duration_minutes,
    _repositioning_duration_minutes,
    DEFAULT_FLEET_SIZE,
    RANDOM_SEED,
    VALID_ZONE_IDS,
    NUM_ZONES,
)

print("=" * 60)
print("1. CAPACITY ASSUMPTION VALIDATION")
print("=" * 60)

capacity = _capacity_per_cab_per_interval()
trip_cycle = _trip_cycle_duration_minutes()
reposition_dur = _repositioning_duration_minutes()
print(f"Trip cycle duration: {trip_cycle} min")
print(f"Repositioning duration: {reposition_dur} min")
print(f"Capacity per cab per interval: {capacity} rides")
print(f"Expected: (60 min interval) / (35+10=45 min cycle) = {60/45:.2f} rides")
print(f"Match: {abs(capacity - 60/45) < 0.01}")

print()
print("=" * 60)
print("2. HAND-CALCULATED TEST SCENARIOS")
print("=" * 60)

# Use a 3-zone mini-state by modifying the proportions
state = _make_initial_state(fleet_size=30)  # small fleet for testing
print(f"\n3-zone fleet (size=30):")
for zid in state['zones'][:3]:
    idle = state['idle_cabs'].get(zid, 0)
    busy = state['busy_cabs'].get(zid, 0)
    repos = state['repositioning_cabs'].get(zid, 0)
    total = state['total_cabs_by_zone'].get(zid, 0)
    print(f"  Zone {zid}: idle={idle}, busy={busy}, repos={repos}, total={total}, sum={idle+busy+repos}")

# Test zero demand
# Test zero demand
print()
print("=== Zero Demand Test ===")

state_zero = _make_initial_state(fleet_size=30)
state_zero["predicted_rides"] = {
    zid: 0 for zid in state_zero["zones"]
}

rec = compute_allocation_recommendations(state_zero)

print(f"Unmet demand: {rec['unmet_demand_estimate']}")
print(f"Recommendations: {len(rec['recommended_movements'])}")

unmet = rec["unmet_demand_estimate"]

assert (
    not unmet or unmet.get("total_unmet_rides", 0) == 0
), "Zero demand must not produce unmet rides"
assert len(rec["recommended_movements"]) == 0, (
    "Zero demand must not produce demand-driven repositioning"
)

print("PASS: Zero demand produces no unmet rides or recommendations.")
# Test zero cabs
print()
print("=== Zero Available Cabs Test ===")
state_no_cabs = _make_initial_state(fleet_size=0)
print(f"fleet_size={state_no_cabs['fleet_size']}")
rec_no_cabs = compute_allocation_recommendations(state_no_cabs)
print(f"Unmet demand: {rec_no_cabs['unmet_demand_estimate']}")
print(f"Zone metrics shortage: {[m['shortage_estimate'] for m in rec_no_cabs['zones'][:3]]}")

# Test demand > capacity
print()
print("=== Demand > Capacity Test ===")
state_heavy = _make_initial_state(fleet_size=30)
rec_heavy = compute_allocation_recommendations(state_heavy)
print(f"Unmet demand: {rec_heavy['unmet_demand_estimate']}")
print(f"Zone metrics shortage (first 3): {[m['shortage_estimate'] for m in rec_heavy['zones'][:3]]}")
print(f"Zone metrics surplus (first 3): {[m['surplus'] for m in rec_heavy['zones'][:3]]}")

print()
print("=" * 60)
print("3. REPOSITIONING RECOMMENDATIONS - DETERMINISTIC TESTS")
print("=" * 60)

# Test with a state that has clear surplus/shortage
state_test = _make_initial_state(fleet_size=50)
rec_test = compute_allocation_recommendations(state_test)
print(f"Recommendations count: {len(rec_test['recommended_movements'])}")
print(f"Unmet demand: {rec_test['unmet_demand_estimate']}")
for m in rec_test['recommended_movements'][:5]:
    print(f"  From zone {m['from_zone']} -> To zone {m['to_zone']}: {m['cab_count']} cabs, reason: {m['reason']}")

# Test no shortage scenario
print()
print("=== No Shortage Scenario ===")
state_no_short = _make_initial_state(fleet_size=30)
rec_no_short = compute_allocation_recommendations(state_no_short)
print(f"Recommendations count: {len(rec_no_short['recommended_movements'])}")
print(f"Unmet demand: {rec_no_short['unmet_demand_estimate']}")

# Test multiple shortage zones
print()
print("=== Multiple Shortage Zones ===")
state_multi = _make_initial_state(fleet_size=100)
rec_multi = compute_allocation_recommendations(state_multi)
print(f"Recommendations count: {len(rec_multi['recommended_movements'])}")
print(f"Unmet demand: {rec_multi['unmet_demand_estimate']}")

print()
print("=" * 60)
print("4. DEMAND ESTIMATION CONSISTENCY")
print("=" * 60)

# Compare main.py init demand vs simulation demand
from fleet_simulation import DEMAND_PROPORTIONS

# main.py sets: predicted_rides = proportions * (fleet_size // 10)
fleet_size = 13000
total_predicted = fleet_size // 10  # = 1300
print(f"main.py init demand: total_predicted = {total_predicted}")
print(f"Sum of DEMAND_PROPORTIONS: {sum(DEMAND_PROPORTIONS.values()):.4f}")

# _predict_demand_for_interval uses historical data
# Let's check what it produces for a small fleet
state_small = _make_initial_state(fleet_size=50)
rec_heavy = compute_allocation_recommendations(state_small)
print(f"\nSmall fleet (size=50) recommendations unmet: {rec_heavy['unmet_demand_estimate']['total_unmet_rides']}")
print("Zone metrics (first 3):")
for m in rec_heavy['zones'][:3]:
    print(f"  Zone {m['zone_id']}: predicted={m['predicted_rides']}, idle={m['idle_cabs']}, required={m['estimated_cabs_required']}, surplus={m['surplus']}, shortage={m['shortage_estimate']}")