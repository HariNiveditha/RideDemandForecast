"""High-level forecasting service exposing unified single-zone and batch prediction interfaces."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from forecasting.config import (
    ARTIFACTS_DIR,
    DEFAULT_HORIZONS_MINUTES,
    HORIZON_STEPS,
    MODEL_VERSION,
    MULTI_HORIZON_FEATURE_ORDER,
    WEATHER_FEATURES,
)
from forecasting.data import build_regular_hourly_grid, load_raw_dataset, load_zone_lookup
from forecasting.layer3_contract import ForecastRecord, validate_forecast_records
from forecasting.models import MultiHorizonDemandModel
from forecasting.uncertainty import ConformalResidualCalibrator

logger = logging.getLogger(__name__)


class ForecastingService:
    """Production-ready forecasting service for Layer 1 multi-horizon demand forecasting."""

    _instance: Optional["ForecastingService"] = None

    def __init__(
        self,
        model: Optional[MultiHorizonDemandModel] = None,
        calibrator: Optional[ConformalResidualCalibrator] = None,
        grid_df: Optional[pd.DataFrame] = None,
        weather_df: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.calibrator = calibrator or ConformalResidualCalibrator()
        self.grid_df = grid_df
        self.weather_df = weather_df
        self.zone_lookup = load_zone_lookup()
        self.known_zones = set(self.zone_lookup["PULocationID"].unique())
        self._history_series: Optional[pd.Series] = None
        self._global_hourly_profile: Dict[Tuple[int, int], float] = {}

    @classmethod
    def get_default_service(cls) -> "ForecastingService":
        """Singleton accessor for default forecasting service."""
        if cls._instance is None:
            service = cls()
            service.initialize()
            cls._instance = service
        return cls._instance

    def initialize(self) -> None:
        """Load datasets, regular grid, trained models, and uncertainty calibrator."""
        logger.info("Initializing ForecastingService...")
        raw_df = load_raw_dataset()
        self.grid_df, self.weather_df = build_regular_hourly_grid(raw_df)
        self._history_series = (
            self.grid_df.set_index(["PULocationID", "pickup_hour"])["ride_count"].sort_index()
        )

        # Precompute zone x dow x hour profile for robust fallback on sparse/missing history
        self.grid_df["hour"] = self.grid_df["pickup_hour"].dt.hour
        self.grid_df["dow"] = self.grid_df["pickup_hour"].dt.dayofweek
        grouped_profile = self.grid_df.groupby(["PULocationID", "dow", "hour"])["ride_count"].mean()
        self._global_hourly_profile = grouped_profile.to_dict()

        # Load trained models
        try:
            self.model = MultiHorizonDemandModel.load(ARTIFACTS_DIR)
            self.calibrator = ConformalResidualCalibrator.load(ARTIFACTS_DIR)
            logger.info("ForecastingService successfully loaded pre-trained artifacts.")
        except Exception as e:
            logger.warning(f"Could not load pre-trained artifacts ({e}). Training will be required.")

    def _get_historical_demand(self, zone_id: int, timestamp: pd.Timestamp) -> float:
        """Lookup historical demand for (zone_id, timestamp), defaulting to 0.0 if not present."""
        if self._history_series is None:
            return 0.0
        try:
            val = self._history_series.loc[(zone_id, timestamp)]
            if isinstance(val, pd.Series):
                return float(val.iloc[0])
            return float(val)
        except KeyError:
            # Missing observation in quiet or unobserved periods
            return 0.0

    def _build_features_for_zone(
        self,
        zone_id: int,
        origin_dt: pd.Timestamp,
        horizon_minutes: int,
        weather_overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Construct leak-free feature vector for a specific zone and horizon as of origin_dt."""
        step = HORIZON_STEPS[horizon_minutes]
        target_dt = origin_dt + pd.Timedelta(hours=step)

        # Lags available at origin_dt
        demand_0 = self._get_historical_demand(zone_id, origin_dt)
        demand_1 = self._get_historical_demand(zone_id, origin_dt - pd.Timedelta(hours=1))

        # Target-aligned lags: (target_dt - 24h) and (target_dt - 168h)
        target_lag_24_time = target_dt - pd.Timedelta(hours=24)
        target_lag_168_time = target_dt - pd.Timedelta(hours=168)
        target_lag_24 = self._get_historical_demand(zone_id, target_lag_24_time)
        target_lag_168 = self._get_historical_demand(zone_id, target_lag_168_time)

        # 24h rolling window up to origin_dt
        rolling_hours = [origin_dt - pd.Timedelta(hours=i) for i in range(24)]
        rolling_values = [self._get_historical_demand(zone_id, t) for t in rolling_hours]
        rolling_mean_24 = float(np.mean(rolling_values)) if rolling_values else 0.0
        rolling_std_24 = float(np.std(rolling_values)) if len(rolling_values) > 1 else 0.0

        # Target calendar features
        hour = target_dt.hour
        dow = target_dt.dayofweek
        cal_features = {
            "target_lag_168": target_lag_168,
            "target_lag_24": target_lag_24,
            "origin_lag_0": demand_0,
            "origin_lag_1": demand_1,
            "origin_rolling_mean_24": rolling_mean_24,
            "origin_rolling_std_24": rolling_std_24,
            "PULocationID": int(zone_id),
            "hour_of_day": hour,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dayofweek": dow,
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "is_weekend": int(dow in (5, 6)),
            "day": target_dt.day,
            "month": target_dt.month,
            "weekofyear": int(target_dt.isocalendar().week),
            "quarter": target_dt.quarter,
            "dayofyear": target_dt.dayofyear,
        }

        # Target weather features
        weather_dict = {}
        if self.weather_df is not None and target_dt in self.weather_df.index:
            w_row = self.weather_df.loc[target_dt]
            for col in WEATHER_FEATURES:
                weather_dict[f"weather_{col}"] = float(w_row.get(col, 0.0))
        else:
            # Fallback default weather
            default_weather = {
                "weather_temp": 68.0,
                "weather_feelslike": 68.0,
                "weather_humidity": 55.0,
                "weather_precip": 0.0,
                "weather_precipprob": 0.0,
                "weather_snow": 0.0,
                "weather_windgust": 5.0,
                "weather_windspeed": 7.0,
                "weather_sealevelpressure": 1015.0,
                "weather_cloudcover": 20.0,
            }
            weather_dict.update(default_weather)

        # Apply any caller-specified weather overrides
        if weather_overrides:
            for k, v in weather_overrides.items():
                weather_key = k if k.startswith("weather_") else f"weather_{k}"
                weather_dict[weather_key] = float(v)

        cal_features.update(weather_dict)
        return pd.DataFrame([cal_features])[MULTI_HORIZON_FEATURE_ORDER]

    def predict_zone(
        self,
        zone_id: int,
        forecast_origin: Union[str, datetime, pd.Timestamp],
        horizon_minutes: int = 60,
        weather_overrides: Optional[Dict[str, float]] = None,
        include_bounds: bool = True,
    ) -> Dict[str, Any]:
        """Generate demand forecast for a single taxi zone and horizon.
        
        Returns record strictly complying with Layer 3 integration contract.
        """
        if horizon_minutes not in HORIZON_STEPS:
            raise ValueError(
                f"Unsupported horizon_minutes: {horizon_minutes}. "
                f"Supported horizons: {list(HORIZON_STEPS.keys())}"
            )

        origin_dt = pd.Timestamp(forecast_origin).floor("h")
        if origin_dt.tzinfo is not None:
            origin_dt = origin_dt.tz_convert(None)

        step = HORIZON_STEPS[horizon_minutes]
        target_dt = origin_dt + pd.Timedelta(hours=step)

        # Handle unknown/unobserved zones or missing models
        if self.model is None or horizon_minutes not in self.model.models:
            # Fallback to historical profile
            fallback_val = self._global_hourly_profile.get(
                (int(zone_id), target_dt.dayofweek, target_dt.hour), 0.0
            )
            point_pred = max(0.0, float(fallback_val))
            lower_bound = max(0.0, point_pred * 0.7) if include_bounds else None
            upper_bound = max(point_pred, point_pred * 1.3) if include_bounds else None
        else:
            X = self._build_features_for_zone(zone_id, origin_dt, horizon_minutes, weather_overrides)
            point_pred = float(self.model.predict(X, horizon_minutes)[0])
            point_pred = round(max(0.0, point_pred), 2)

            if include_bounds:
                l, u = self.calibrator.predict_bounds(point_pred, horizon_minutes)
                lower_bound = round(float(l), 2)
                upper_bound = round(float(u), 2)
            else:
                lower_bound = None
                upper_bound = None

        record = {
            "zone_id": int(zone_id),
            "forecast_origin": origin_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "target_timestamp": target_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "horizon_minutes": int(horizon_minutes),
            "predicted_rides": point_pred,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model_version": MODEL_VERSION,
        }

        # Validate against contract
        validated = ForecastRecord(**record)
        return validated.model_dump()

    def forecast_batch(
        self,
        zone_ids: Optional[List[int]] = None,
        forecast_origin: Optional[Union[str, datetime, pd.Timestamp]] = None,
        horizons_minutes: Optional[List[int]] = None,
        weather_overrides: Optional[Dict[str, float]] = None,
        include_bounds: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batch forecast across multiple taxi zones and horizons."""
        target_zones = zone_ids or sorted(list(self.known_zones))
        target_horizons = horizons_minutes or DEFAULT_HORIZONS_MINUTES
        origin = forecast_origin or (self.grid_df["pickup_hour"].max() if self.grid_df is not None else "2021-06-30T23:00:00")

        results = []
        for h in target_horizons:
            for zid in target_zones:
                rec = self.predict_zone(
                    zone_id=zid,
                    forecast_origin=origin,
                    horizon_minutes=h,
                    weather_overrides=weather_overrides,
                    include_bounds=include_bounds,
                )
                results.append(rec)
        return results


# Module-level convenience functions
def predict_zone(
    zone_id: int,
    forecast_origin: Union[str, datetime, pd.Timestamp],
    horizon_minutes: int = 60,
    weather_overrides: Optional[Dict[str, float]] = None,
    include_bounds: bool = True,
) -> Dict[str, Any]:
    """Module-level single-zone prediction function."""
    service = ForecastingService.get_default_service()
    return service.predict_zone(
        zone_id=zone_id,
        forecast_origin=forecast_origin,
        horizon_minutes=horizon_minutes,
        weather_overrides=weather_overrides,
        include_bounds=include_bounds,
    )


def forecast_batch(
    zone_ids: Optional[List[int]] = None,
    forecast_origin: Optional[Union[str, datetime, pd.Timestamp]] = None,
    horizons_minutes: Optional[List[int]] = None,
    weather_overrides: Optional[Dict[str, float]] = None,
    include_bounds: bool = True,
) -> List[Dict[str, Any]]:
    """Module-level batch forecasting function."""
    service = ForecastingService.get_default_service()
    return service.forecast_batch(
        zone_ids=zone_ids,
        forecast_origin=forecast_origin,
        horizons_minutes=horizons_minutes,
        weather_overrides=weather_overrides,
        include_bounds=include_bounds,
    )
