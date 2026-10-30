"""Integration contract, data models, validation, and export adapters for Layer 3 (Fleet Management)."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from forecasting.config import FIXTURES_DIR, MODEL_VERSION, DEFAULT_MAX_FORECAST_AGE_HOURS

# NOTE: DEFAULT_MAX_FORECAST_AGE_HOURS is defined in forecasting.config and
# re-exported here for Layer 3 contract consumers.


class ForecastContractError(ValueError):
    """Raised when Layer 1 forecast records cannot be consumed by Layer 3.

    The message is safe to surface to API clients: it identifies the offending
    field or zone without leaking unrelated forecast payload.
    """



class ForecastRecord(BaseModel):
    """Canonical Layer 1 output schema representing hourly zone-level ride demand forecasts.

    IMPORTANT INTEGRATION NOTE FOR LAYER 3 (FLEET SUPPLY ESTIMATION):
      `predicted_rides` strictly represents the estimated passenger ride demand (completed pickups)
      originating in `zone_id` over the 1-hour interval starting at `target_timestamp`.
      It DOES NOT represent current fleet supply, available cabs, or required vehicles.
    """
    zone_id: int = Field(..., gt=0, description="NYC Taxi & Limousine Commission official Taxi Zone ID (PULocationID).")
    forecast_origin: str = Field(..., description="ISO 8601 timestamp string (YYYY-MM-DDTHH:MM:SS) representing when the forecast was generated.")
    target_timestamp: str = Field(..., description="ISO 8601 timestamp string representing the start of the 1-hour interval being predicted.")
    horizon_minutes: int = Field(..., gt=0, description="Lead time in minutes from forecast_origin to target_timestamp (e.g. 60, 180, 360).")
    predicted_rides: float = Field(..., ge=0.0, description="Predicted passenger ride demand (expected completed pickups) during the target hour.")
    lower_bound: Optional[float] = Field(None, ge=0.0, description="Calibrated lower prediction bound (e.g. 80% confidence). Clamped at 0.0.")
    upper_bound: Optional[float] = Field(None, ge=0.0, description="Calibrated upper prediction bound (e.g. 80% confidence). >= lower_bound.")
    model_version: Optional[str] = Field(default=MODEL_VERSION, description="Identifier of the forecasting model generating this prediction.")

    @field_validator("upper_bound")
    @classmethod
    def check_bounds(cls, v: Optional[float], info) -> Optional[float]:
        lower = info.data.get("lower_bound")
        if v is not None and lower is not None and v < lower:
            raise ValueError(f"upper_bound ({v}) cannot be less than lower_bound ({lower})")
        return v


def validate_forecast_records(records: List[Dict[str, Any]]) -> List[ForecastRecord]:
    """Validate a list of raw forecast dictionaries against the Layer 3 schema."""
    validated = []
    for item in records:
        validated.append(ForecastRecord(**item))
    return validated


def export_forecasts_to_json(
    records: List[Union[ForecastRecord, Dict[str, Any]]],
    filepath: Union[str, Path],
) -> Path:
    """Save forecast records to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.model_dump() if isinstance(r, ForecastRecord) else r for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def export_forecasts_to_csv(
    records: List[Union[ForecastRecord, Dict[str, Any]]],
    filepath: Union[str, Path],
) -> Path:
    """Save forecast records to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.model_dump() if isinstance(r, ForecastRecord) else r for r in records]
    if not data:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = [
        "zone_id",
        "forecast_origin",
        "target_timestamp",
        "horizon_minutes",
        "predicted_rides",
        "lower_bound",
        "upper_bound",
        "model_version",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def get_layer3_json_schema() -> Dict[str, Any]:
    """Generate the official JSON Schema for Layer 3 integration."""
    schema = ForecastRecord.model_json_schema()
    schema["title"] = "Layer1DemandForecastRecord"
    schema["description"] = (
        "Official data contract for Layer 1 multi-horizon zone-level taxi ride demand forecasting. "
        "Consumable by Layer 3 fleet supply estimation."
    )
    return schema


class Layer3Adapter:
    """Adapter utility for converting Layer 1 forecasts into formats required by fleet simulation."""

    @staticmethod
    def to_dataframe(records: List[Union[ForecastRecord, Dict[str, Any]]]) -> pd.DataFrame:
        """Convert forecast records into a clean pandas DataFrame."""
        data = [r.model_dump() if isinstance(r, ForecastRecord) else r for r in records]
        df = pd.DataFrame(data)
        if not df.empty:
            df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])
            df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
            df["zone_id"] = df["zone_id"].astype(int)
            df["horizon_minutes"] = df["horizon_minutes"].astype(int)
            df["predicted_rides"] = df["predicted_rides"].astype(float)
        return df

    @staticmethod
    def to_zone_matrix(records: List[Union[ForecastRecord, Dict[str, Any]]], horizon_minutes: int) -> Dict[int, float]:
        """Convert forecasts for a specific horizon into a lookup map: {zone_id: predicted_rides}."""
        data = [r.model_dump() if isinstance(r, ForecastRecord) else r for r in records]
        return {
            int(r["zone_id"]): float(r["predicted_rides"])
            for r in data
            if int(r["horizon_minutes"]) == horizon_minutes
        }

    @staticmethod
    def to_fleet_demand(
        records: List[Union[ForecastRecord, Dict[str, Any]]],
        horizon_minutes: int,
        expected_zone_ids: List[int],
        *,
        expected_origin: Optional[Union[str, pd.Timestamp]] = None,
        latest_available_timestamp: Optional[Union[str, pd.Timestamp]] = None,
        max_staleness_hours: Optional[float] = DEFAULT_MAX_FORECAST_AGE_HOURS,
    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
        """Validate Layer 1 forecast records and convert them to fleet demand.

        The returned mapping is ``{zone_id: predicted_rides}`` where
        ``predicted_rides`` is the passenger ride demand during the single
        1-hour interval beginning at ``target_timestamp`` — the same unit and
        interval used by the fleet allocation algorithm
        (``SIMULATION_INTERVAL_HOURS == 1``). Horizon length therefore changes
        *which* hour is forecast, never the demand units.

        Raises ForecastContractError when the records are unusable, covering:
        empty/missing records, schema violations, negative or non-finite
        demand, unsupported horizon, duplicate zones, unknown zones, missing
        zones, inconsistent origins/targets, and stale origins.

        Returns (demand_by_zone, metadata).
        """
        if not records:
            raise ForecastContractError(
                f"No forecast records were returned for horizon_minutes={horizon_minutes}."
            )

        expected_zones = {int(z) for z in expected_zone_ids}
        if not expected_zones:
            raise ForecastContractError("No expected zones were supplied for forecast validation.")

        # 1. Schema validation through the official Layer 3 contract.
        try:
            validated = validate_forecast_records(list(records))
        except ForecastContractError:
            raise
        except Exception as exc:  # pydantic ValidationError and friends
            raise ForecastContractError(
                f"Forecast record failed the Layer 3 schema: {exc}"
            ) from exc

        # 2. Horizon filtering.
        matching = [r for r in validated if int(r.horizon_minutes) == int(horizon_minutes)]
        if not matching:
            seen_horizons = sorted({int(r.horizon_minutes) for r in validated})
            raise ForecastContractError(
                f"No forecast records found for horizon_minutes={horizon_minutes}; "
                f"records contained horizons {seen_horizons}."
            )

        # 3. Numeric sanity. pydantic's ge=0.0 rejects negatives but accepts
        #    +inf, so finiteness is checked explicitly here.
        for record in matching:
            if not math.isfinite(record.predicted_rides) or record.predicted_rides < 0.0:
                raise ForecastContractError(
                    f"Zone {record.zone_id}: predicted_rides must be a finite, "
                    f"non-negative number (received {record.predicted_rides!r})."
                )
            for bound_name in ("lower_bound", "upper_bound"):
                bound = getattr(record, bound_name)
                if bound is not None and not math.isfinite(bound):
                    raise ForecastContractError(
                        f"Zone {record.zone_id}: {bound_name} must be finite."
                    )

        # 4. All records must describe one origin/target interval.
        origins = {r.forecast_origin for r in matching}
        if len(origins) != 1:
            raise ForecastContractError(
                f"Forecast records mix {len(origins)} different forecast_origin values; "
                "a single recommendation run must use one origin."
            )
        origin_raw = matching[0].forecast_origin
        origin_ts = _as_naive_hour(origin_raw)

        targets = {r.target_timestamp for r in matching}
        if len(targets) != 1:
            raise ForecastContractError(
                f"Forecast records mix {len(targets)} different target_timestamp values."
            )
        target_raw = matching[0].target_timestamp
        target_ts = _as_naive_hour(target_raw)

        expected_target = origin_ts + pd.Timedelta(minutes=int(horizon_minutes))
        if target_ts != expected_target:
            raise ForecastContractError(
                f"target_timestamp {target_raw} is inconsistent with forecast_origin "
                f"{origin_raw} at horizon_minutes={horizon_minutes} "
                f"(expected {expected_target.strftime('%Y-%m-%dT%H:%M:%S')})."
            )

        if expected_origin is not None:
            wanted = _as_naive_hour(expected_origin)
            if origin_ts != wanted:
                raise ForecastContractError(
                    f"Forecast records were generated for origin "
                    f"{origin_raw} instead of the requested "
                    f"{wanted.strftime('%Y-%m-%dT%H:%M:%S')}."
                )

        # 5. Staleness: an origin far outside the available data range is
        #    rejected rather than silently consumed.
        if latest_available_timestamp is not None and max_staleness_hours is not None:
            latest_ts = _as_naive_hour(latest_available_timestamp)
            age_hours = (latest_ts - origin_ts).total_seconds() / 3600.0
            if age_hours > max_staleness_hours:
                raise ForecastContractError(
                    f"Stale forecast: origin {origin_raw} is {age_hours:.1f}h older than the "
                    f"latest available data hour {latest_ts.strftime('%Y-%m-%dT%H:%M:%S')} "
                    f"(limit {max_staleness_hours:g}h)."
                )
            if age_hours < 0:
                raise ForecastContractError(
                    f"Forecast origin {origin_raw} is after the latest available data hour "
                    f"{latest_ts.strftime('%Y-%m-%dT%H:%M:%S')}."
                )

        # 6. Zone coverage: duplicates, unknowns and gaps are all rejected.
        by_zone: Dict[int, ForecastRecord] = {}
        for record in matching:
            if record.zone_id in by_zone:
                raise ForecastContractError(
                    f"Duplicate forecast record for zone {record.zone_id}."
                )
            by_zone[record.zone_id] = record

        unknown = sorted(z for z in by_zone if z not in expected_zones)
        if unknown:
            raise ForecastContractError(
                f"Forecast returned zone IDs outside the fleet zone registry: "
                f"{_summarize_ids(unknown)}."
            )

        missing = sorted(z for z in expected_zones if z not in by_zone)
        if missing:
            raise ForecastContractError(
                f"Forecast is missing {len(missing)} required zone(s): "
                f"{_summarize_ids(missing)}."
            )

        demand: Dict[int, float] = {
            zid: float(record.predicted_rides) for zid, record in sorted(by_zone.items())
        }
        metadata: Dict[str, Any] = {
            "horizon_minutes": int(horizon_minutes),
            "forecast_origin": origin_raw,
            "target_timestamp": target_raw,
            "model_version": matching[0].model_version,
            "zone_count": len(demand),
            "total_predicted_rides": round(sum(demand.values()), 3),
        }
        return demand, metadata


def _as_naive_hour(value: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """Parse a timestamp and floor it to the hour without timezone info."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.floor("h")


def _summarize_ids(ids: List[int], limit: int = 10) -> str:
    """Render a bounded list of zone IDs for error messages."""
    shown = ", ".join(str(i) for i in ids[:limit])
    if len(ids) > limit:
        shown += f", ... (+{len(ids) - limit} more)"
    return shown
