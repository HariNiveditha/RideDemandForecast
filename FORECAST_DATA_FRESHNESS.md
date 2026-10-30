# Forecast Data Freshness & Timestamp Handling

This note documents how forecast origin timestamps are chosen, why the current
system time is deliberately **not** used, and what must be added to support
genuinely current forecasts.

## 1. The problem

The demand dataset (`ride_demand_weather_hourly.parquet`) ends at
**`2021-06-30T23:00:00`**. When no explicit origin is supplied, the 60 / 180 /
360-minute forecasts are therefore measured from that historical hour:

| Horizon | Meaning | Default target |
|---|---|---|
| 60 min | origin + 1 hour | `2021-07-01T00:00:00` |
| 180 min | origin + 3 hours | `2021-07-01T02:00:00` |
| 360 min | origin + 6 hours | `2021-07-01T05:00:00` |

These are *dataset-relative* forecasts, not forecasts for "right now".

## 2. How the origin is selected (code trace)

1. **`POST /fleet/recommendations`** → `main._resolve_forecast_origin()`
   - `forecast_origin` omitted → `_forecasting_dataset_latest_hour(service)`,
     i.e. `ForecastingService.grid_df["pickup_hour"].max()` floored to the hour.
   - `forecast_origin` supplied → floored to the hour and range-checked against
     `[latest − 24h, latest]` (`forecasting.config.DEFAULT_MAX_FORECAST_AGE_HOURS`).
     Outside that window → **HTTP 400** (`stale forecast_origin` / origin after
     the newest data hour).
2. **`POST /forecast/multi-horizon`** → the caller *must* supply
   `forecast_origin`. The Forecast page restricts its date input to the dataset
   window and rejects anything outside it client-side; the response now reports
   `origin_within_dataset_range` so an out-of-range origin is visible rather
   than silent.
3. **`ForecastingService.forecast_batch(zone_ids, origin, horizons)`** →
   `target_timestamp = forecast_origin + horizon_minutes`. Horizon semantics are
   unchanged and identical on every path.

The system clock (`datetime.now(timezone.utc)`) is used **only** for reporting
in `demand_source.system_time`. It never enters feature construction.

## 3. Is there a live/recent data feed?

**No.** Verified by repository-wide search:

- No upload, ingest, push, or refresh endpoint exists (`main.py` exposes 18
  read/predict endpoints; none accept data).
- `forecasting/data.py::load_raw_dataset` and `main._load_history()` both read
  the same static parquet once and cache it in memory.
- There is no scheduler, no file watcher, no external API client.

Because there is no feed, substituting the current time for the dataset time
would be actively misleading: the model's lag (`lag_1h/24h/168h`) and rolling
features are looked up from the historical grid, so a 2026 origin would resolve
to absent/zero lags and yield a confident-looking but meaningless number.

## 4. What was implemented instead

Every demand-bearing response now discloses its data basis rather than
pretending to be live.

`POST /fleet/recommendations` → `demand_source` (all three paths: `historical`,
`forecast`, `historical_fallback`):

```json
{
  "data_basis": "historical_dataset",
  "dataset_latest_timestamp": "2021-06-30T23:00:00",
  "system_time": "2026-10-30T14:20:00Z",
  "is_live": false,
  "freshness_note": "Data is anchored to the dataset's latest available hour (...) not to the current system time (...). No live demand feed is configured, so horizons are dataset-relative rather than live-anchored.",
  "origin_is_dataset_latest": true,
  "target_beyond_dataset_end": true
}
```

`GET /forecast/horizons` additionally returns `horizon_reference` plus the same
freshness block. `POST /forecast/multi-horizon` returns the freshness block plus
`origin_is_dataset_latest` and `origin_within_dataset_range`.

The Fleet Management page renders this as:

> **Data basis:** historical dataset (not live); latest demand data
> 2021-06-30T23:00:00; response generated 2026-10-30T14:20:00Z.

No demand value, timestamp, or accuracy figure is fabricated anywhere: the
dataset timestamp and the system timestamp are reported side by side and are
never interchanged.

## 5. What is needed for genuinely current forecasts

The integration point is deliberately narrow — **no horizon handling or
allocation code has to change.**

1. **Ingest.** Add an hourly (or more frequent) job that appends real pickup
   counts for zones/hours *after* `2021-06-30T23:00:00` into
   `ride_demand_weather_hourly.parquet` (same schema: `pickup_hour`,
   `PULocationID`, `ride_count`, plus the weather columns in
   `forecasting.config.WEATHER_FEATURES`).
2. **Expose it.** Either rewrite the parquet in place, or add a data-loading
   hook that `main._load_history()` and `ForecastingService.grid_df` read from.
   Both are already lazy and idempotent, so a cache-busting reload function is
   the only new surface required.
3. **Freshness follows automatically.** Once the dataset advances,
   `_forecasting_dataset_latest_hour()` and `_dataset_range()` return the new
   hour, so:
   - the default forecast origin moves forward on its own,
   - the 24-hour staleness window keeps rejecting genuinely stale origins,
   - `dataset_latest_timestamp` and `is_live` reflect reality with no code edit
     beyond flipping `data_basis`/`is_live` once a feed is confirmed.
4. **Weather.** The model also consumes weather features for the target hour;
   a current forecast needs a matching weather source (or the existing
   `weather_overrides` on `/forecast/multi-horizon`).
5. **Then** update `FORECAST_DATA_FRESHNESS.md`, `_freshness_metadata()` and the
   UI copy, which currently state that no live feed exists.

Until step 1–2 are in place, `data_basis` stays `"historical_dataset"` and
`is_live` stays `false`.

## 6. Guarantees

- Historical allocation remains the **default** for `/fleet/recommendations`
  (no body / no horizon), unchanged.
- The explicit `allow_fallback` contract is unchanged: forecasting failures
  either return an error or an explicitly flagged `historical_fallback`.
- Horizon meanings (60 / 180 / 360 minutes from the origin) are unchanged on
  every path.
- Existing API fields are only **added to**, never removed or renamed.
