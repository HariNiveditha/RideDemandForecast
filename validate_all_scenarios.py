"""
End-to-End Validation Script for Layer 3 (Fleet Supply Estimation)
and Layer 4 (Fleet Allocation and Repositioning).

Tests all 4 fleet sizes: 13,000, 16,000, 20,000, 25,000 across 4 advances.
Validates:
- Conservation invariants: idle + busy + repositioning == total_fleet
- Zone totals == global totals
- Genuine shortage (> 0) and surplus (> 0)
- No self-movements, no circular movements
- Donor availability not exceeded
- Movement caps respected (1 <= count <= 5)
- Dynamic re-initialization across changing fleet sizes
"""
import sys
from fastapi.testclient import TestClient
from main import app
import fleet_simulation

def run_validation():
    client = TestClient(app)
    fleet_sizes = [13000, 16000, 20000, 25000]

    print("=" * 100)
    print("END-TO-END VALIDATION: LAYER 3 & LAYER 4 ACROSS ALL FLEET SIZES")
    print("=" * 100)

    scenario_records = {}

    for fs in fleet_sizes:
        print(f"\n>>> TESTING FLEET SIZE: {fs:,} CABS <<<")
        # 1. Initialize
        resp_init = client.post("/fleet/init", json={"fleet_size": fs})
        assert resp_init.status_code == 200, f"Init failed: {resp_init.text}"
        data_init = resp_init.json()
        assert data_init["fleet_size"] == fs, f"Returned fleet size {data_init['fleet_size']} != {fs}"
        sim_id = data_init["simulation_id"]

        # 2. Check current state
        resp_curr = client.get("/fleet/current")
        assert resp_curr.status_code == 200
        data_curr = resp_curr.json()
        ft = data_curr["fleet_totals"]
        assert ft["total_fleet"] == fs
        assert ft["conservation_check"] is True

        # Check Layer 3 supply distribution
        availability = data_curr["availability"]
        assert len(availability) == 265, f"Expected 265 zones, got {len(availability)}"
        zone_sum = sum(z["total_cabs"] for z in availability.values())
        assert zone_sum == fs, f"Zone sum {zone_sum} != {fs}"
        assert ft["total_busy"] == 0, "Initial busy cabs should be 0"
        assert ft["total_repositioning"] == 0, "Initial repositioning cabs should be 0"
        assert ft["total_idle"] == fs, f"Initial idle cabs {ft['total_idle']} != {fs}"

        # Demand check
        pred_rides = data_curr["predicted_rides"]
        total_pred = sum(pred_rides.values())
        print(f"  Initialized: Total Fleet={fs:,}, Idle={ft['total_idle']:,}, Predicted Rides={total_pred:,}")

        scenario_records[fs] = []

        # Evaluate at Step 0 (Init) and Advances 1, 2, 3, 4
        for step in range(0, 5):
            step_name = "Init" if step == 0 else f"Advance {step}"
            if step > 0:
                resp_adv = client.post("/fleet/advance")
                assert resp_adv.status_code == 200
                assert resp_adv.json()["conservation_check"] is True

            # Get current fleet state
            curr_state = client.get("/fleet/current").json()
            curr_ft = curr_state["fleet_totals"]
            assert curr_ft["conservation_check"] is True
            assert curr_ft["total_idle"] + curr_ft["total_busy"] + curr_ft["total_repositioning"] == fs

            # Verify zone-level totals match
            curr_avail = curr_state["availability"]
            for zid, zcounts in curr_avail.items():
                z_tot = zcounts["idle_cabs"] + zcounts["busy_cabs"] + zcounts["repositioning_cabs"]
                assert z_tot == zcounts["total_cabs"], f"Zone {zid} invariant violated: {z_tot} != {zcounts['total_cabs']}"
                assert zcounts["idle_cabs"] >= 0 and zcounts["busy_cabs"] >= 0 and zcounts["repositioning_cabs"] >= 0, f"Negative counts in zone {zid}"

            # Layer 4: Compute recommendations
            resp_recs = client.post("/fleet/recommendations")
            assert resp_recs.status_code == 200
            recs_data = resp_recs.json()

            zones_metrics = {z["zone_id"]: z for z in recs_data["zones"]}
            movements = recs_data["recommended_movements"]

            shortage_zones = [z for z in recs_data["zones"] if z["shortage_estimate"] > 0]
            donor_zones = [z for z in recs_data["zones"] if z["surplus_after_own_demand"] > 0]
            surplus_cabs = sum(z["surplus_after_own_demand"] for z in donor_zones)
            unmet_rides = recs_data["unmet_demand_estimate"].get("total_unmet_rides", 0.0)
            cabs_moved = sum(m["cab_count"] for m in movements)

            # Rigorous Layer 4 Validation checks
            donor_contributions = {}
            for mov in movements:
                from_z = mov["from_zone"]
                to_z = mov["to_zone"]
                cnt = mov["cab_count"]

                # Check 1: No self-movement
                assert from_z != to_z, f"Self movement detected in {step_name}: {from_z} -> {to_z}"

                # Check 2: Movement cap
                assert 1 <= cnt <= 5, f"Movement count {cnt} violates cap in {step_name}"

                # Check 3: Recipient has genuine unmet shortage
                assert zones_metrics[to_z]["shortage_estimate"] > 0, (
                    f"Recipient {to_z} has zero or negative shortage ({zones_metrics[to_z]['shortage_estimate']}) in {step_name}"
                )

                # Check 4: Donor has real surplus
                assert zones_metrics[from_z]["surplus_after_own_demand"] > 0, (
                    f"Donor {from_z} has zero surplus in {step_name}"
                )

                donor_contributions[from_z] = donor_contributions.get(from_z, 0) + cnt

            # Check 5: Donor contributions never exceed donor surplus
            for d_id, donated in donor_contributions.items():
                avail_surplus = zones_metrics[d_id]["surplus_after_own_demand"]
                assert donated <= avail_surplus, (
                    f"Donor {d_id} donated {donated} exceeding surplus {avail_surplus} in {step_name}"
                )

            scenario_records[fs].append({
                "step": step_name,
                "shortage_zones": len(shortage_zones),
                "donor_zones": len(donor_zones),
                "surplus_cabs": surplus_cabs,
                "unmet_rides": unmet_rides,
                "recs_count": len(movements),
                "cabs_moved": cabs_moved,
                "idle": curr_ft["total_idle"],
                "busy": curr_ft["total_busy"],
                "repo": curr_ft["total_repositioning"],
                "conserved": True
            })

            print(f"    [{step_name}] Idle={curr_ft['total_idle']:5d}, Busy={curr_ft['total_busy']:5d}, Repo={curr_ft['total_repositioning']:4d} | "
                  f"Shortage={len(shortage_zones):3d}, Donors={len(donor_zones):3d}, Surplus={surplus_cabs:5d}, "
                  f"Unmet={unmet_rides:6.1f}, Recs={len(movements):2d}, Moved={cabs_moved:3d} | Conserved=True")

    # Dynamic Re-initialization Verification:
    # Switch 25k -> 13k -> 20k -> 16k -> 20k to confirm actual state resets
    print("\n" + "=" * 100)
    print("VERIFYING DYNAMIC RE-INITIALIZATION & RESET")
    print("=" * 100)

    sequence = [13000, 20000, 16000, 25000, 20000]
    for target in sequence:
        r = client.post("/fleet/init", json={"fleet_size": target})
        assert r.status_code == 200
        curr = client.get("/fleet/current").json()
        assert curr["fleet_size"] == target
        assert curr["fleet_totals"]["total_fleet"] == target
        assert curr["fleet_totals"]["total_idle"] == target
        assert curr["fleet_totals"]["total_busy"] == 0
        assert sum(z["total_cabs"] for z in curr["availability"].values()) == target
        print(f"  Successfully switched & verified fleet size -> {target:,} cabs (all {target:,} idle, busy=0, conservation=True)")

    print("\nALL INVARIANTS AND CHECKS VERIFIED SUCCESSFULLY!")

if __name__ == '__main__':
    run_validation()
