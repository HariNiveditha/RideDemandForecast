import sys
sys.path.insert(0, r'C:\Ride Demand Forecasting')

from fleet_simulation import _make_initial_state

state = _make_initial_state()
print(f'fleet_size={state["fleet_size"]}')
print(f'zones={state["zones"][:3]}...')
print(f'conservation: {sum(state["total_cabs_by_zone"].values())} == {state["fleet_size"]}')