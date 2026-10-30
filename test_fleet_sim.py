import sys
from pathlib import Path

# Import the fleet_simulation module that lives beside this script so the
# check always runs against the current working tree, not a sibling checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fleet_simulation import _make_initial_state

state = _make_initial_state()
print(f'fleet_size={state["fleet_size"]}')
print(f'zones={state["zones"][:3]}...')
print(f'conservation: {sum(state["total_cabs_by_zone"].values())} == {state["fleet_size"]}')
