"""Integration contract, data models, validation, and export adapters for Layer 3 (Fleet Management)."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from forecasting.config import FIXTURES_DIR, MODEL_VERSION


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
