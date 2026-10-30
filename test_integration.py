import urllib.request
import json

base = 'http://localhost:8000'

endpoints = [
    ('/metrics', 'GET'),
    ('/locations', 'GET'),
    ('/predictions-sample', 'GET'),
    ('/historical', 'GET'),
    ('/docs', 'GET'),
]

print("=" * 60)
print("Existing endpoints:")
for path, method in endpoints:
    try:
        r = urllib.request.urlopen(f'{base}{path}', timeout=3)
        print(f"  {method} {path}: Status {r.status} OK")
    except Exception as e:
        print(f"  {method} {path}: Error - {e}")

print("\n" + "=" * 60)
print("Fleet endpoints:")

# Fleet init
print("\n  POST /fleet/init")
req = urllib.request.Request(
    f'{base}/fleet/init',
    data=json.dumps({}).encode(),
    method='POST',
    headers={'Content-Type': 'application/json'}
)
try:
    r = urllib.request.urlopen(req, timeout=3)
    init_data = json.loads(r.read().decode())
    print(f"    Status: {r.status}, fleet_size: {init_data.get('fleet_size')}, simulation_id: {init_data.get('simulation_id')}")
except Exception as e:
    print(f"    Error: {e}")

# Fleet current
print("\n  GET /fleet/current")
try:
    r = urllib.request.urlopen(f'{base}/fleet/current', timeout=3)
    current_data = json.loads(r.read().decode())
    availability = current_data.get('availability', {})
    total_cabs = sum(z.get('total_cabs', 0) for z in availability.values())
    print(f"    Status: {r.status}, simulation_id: {current_data.get('simulation_id')}")
    print(f"    Zones: {len(availability)}, total cabs: {total_cabs}")
    first_zone = list(availability.keys())[0] if availability else None
    if first_zone:
        print(f"    Zone {first_zone}: {availability[first_zone]}")
except Exception as e:
    print(f"    Error: {e}")

# Fleet recommendations
print("\n  POST /fleet/recommendations")
try:
    req = urllib.request.Request(f'{base}/fleet/recommendations', method='POST')
    r = urllib.request.urlopen(req, timeout=3)
    rec_data = json.loads(r.read().decode())
    zones = rec_data.get('zones', [])
    print(f"    Status: {r.status}, zones: {len(zones)}")
    if zones:
        print(f"    First zone: zone_id={zones[0].get('zone_id')}, predicted_rides={zones[0].get('predicted_rides')}, shortage={zones[0].get('shortage_estimate')}")
except Exception as e:
    print(f"    Error: {e}")

# Fleet advance
print("\n  POST /fleet/advance")
try:
    req = urllib.request.Request(f'{base}/fleet/advance', method='POST')
    r = urllib.request.urlopen(req, timeout=3)
    advance_data = json.loads(r.read().decode())
    print(f"    Status: {r.status}, conservation_check: {advance_data.get('conservation_check')}")
except Exception as e:
    print(f"    Error: {e}")

# Fleet reset
print("\n  POST /fleet/reset")
try:
    req = urllib.request.Request(f'{base}/fleet/reset', method='POST')
    r = urllib.request.urlopen(req, timeout=3)
    reset_data = json.loads(r.read().decode())
    print(f"    Status: {r.status}, simulation_id: {reset_data.get('simulation_id')}")
except Exception as e:
    print(f"    Error: {e}")

# Fleet export
print("\n  GET /fleet/export")
try:
    r = urllib.request.urlopen(f'{base}/fleet/export', timeout=3)
    export_data = json.loads(r.read().decode())
    print(f"    Status: {r.status}, snapshots: {export_data.get('snapshot_count')}")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 60)
print("All endpoint tests complete!")