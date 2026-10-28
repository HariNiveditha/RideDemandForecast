"""Demand anomaly detection and classification using robust, seasonality-aware methods."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from layer2.config import Layer2Config
from layer2.schemas import AnomalyRecord, AnomalySeverity, AnomalyType


class DemandAnomalyDetector:
    """Robust anomaly detector for zone-level hourly taxi demand."""

    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        # Seasonal profiles: (zone_id, dow, hour) -> (median, mad, count)
        self.seasonal_profiles: Dict[Tuple[int, int, int], Tuple[float, float, int]] = {}
        # Zone fallback profiles: zone_id -> (median, mad)
        self.zone_profiles: Dict[int, Tuple[float, float]] = {}
        self.zone_lookup: Dict[int, Dict[str, str]] = {}

    def fit_seasonal_profile(self, df: pd.DataFrame) -> DemandAnomalyDetector:
        """Fit seasonal median and MAD profiles from historical baseline data (e.g. April-May).
        
        Requires columns: ['pickup_hour' (or 'dayofweek', 'hour_of_day'), 'PULocationID', 'ride_count']
        """
        frame = df.copy()
        if "pickup_hour" in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame["pickup_hour"]):
            frame["pickup_hour"] = pd.to_datetime(frame["pickup_hour"])

        if "dayofweek" not in frame.columns and "pickup_hour" in frame.columns:
            frame["dayofweek"] = frame["pickup_hour"].dt.dayofweek
        if "hour_of_day" not in frame.columns and "pickup_hour" in frame.columns:
            frame["hour_of_day"] = frame["pickup_hour"].dt.hour

        demand_col = "ride_count" if "ride_count" in frame.columns else "actual"

        # Group by (PULocationID, dayofweek, hour_of_day)
        grouped = frame.groupby(["PULocationID", "dayofweek", "hour_of_day"])[demand_col]

        medians = grouped.median()
        # Compute MAD: median of absolute deviations from group median
        frame["group_median"] = grouped.transform("median")
        frame["abs_dev"] = (frame[demand_col] - frame["group_median"]).abs()
        mads = frame.groupby(["PULocationID", "dayofweek", "hour_of_day"])["abs_dev"].median()
        counts = grouped.count()

        self.seasonal_profiles = {}
        for key in medians.index:
            med = float(medians.loc[key])
            mad = float(mads.loc[key])
            cnt = int(counts.loc[key])
            self.seasonal_profiles[key] = (med, mad, cnt)

        # Zone overall profiles for fallback
        z_grouped = frame.groupby("PULocationID")[demand_col]
        z_medians = z_grouped.median()
        frame["z_med"] = z_grouped.transform("median")
        frame["z_abs_dev"] = (frame[demand_col] - frame["z_med"]).abs()
        z_mads = frame.groupby("PULocationID")["z_abs_dev"].median()

        self.zone_profiles = {}
        for zid in z_medians.index:
            self.zone_profiles[int(zid)] = (float(z_medians.loc[zid]), float(z_mads.loc[zid]))

        return self

    def set_zone_lookup(self, lookup: Dict[int, Dict[str, str]]) -> None:
        """Store lookup table for zone names and boroughs."""
        self.zone_lookup = lookup

    def _get_baseline_stats(self, zone_id: int, dow: int, hour: int) -> Tuple[float, float]:
        """Retrieve expected median and MAD for a zone at a specific day-of-week and hour."""
        key = (int(zone_id), int(dow), int(hour))
        if key in self.seasonal_profiles:
            med, mad, _ = self.seasonal_profiles[key]
            return med, mad

        if int(zone_id) in self.zone_profiles:
            return self.zone_profiles[int(zone_id)]

        return 0.0, 1.0

    def calculate_robust_z_score(self, actual: float, median: float, mad: float) -> float:
        """Compute robust z-score: (actual - median) / max(1.4826 * MAD, 1.0)."""
        scale = max(self.config.MAD_SCALE_FACTOR * mad, self.config.EPSILON_STD)
        return (actual - median) / scale

    def determine_severity(
        self,
        z_score: Optional[float],
        is_data_quality: bool = False,
        is_persistent: bool = False,
        consecutive_count: int = 1,
    ) -> AnomalySeverity:
        """Deterministic severity scoring based on robust z-score and persistence."""
        if is_data_quality:
            return AnomalySeverity.HIGH

        if consecutive_count >= 4 or (z_score is not None and abs(z_score) >= self.config.Z_THRESHOLD_CRITICAL):
            return AnomalySeverity.CRITICAL

        if is_persistent or (z_score is not None and abs(z_score) >= self.config.Z_THRESHOLD_HIGH):
            return AnomalySeverity.HIGH

        if z_score is not None and abs(z_score) >= self.config.Z_THRESHOLD_MEDIUM:
            return AnomalySeverity.MEDIUM

        return AnomalySeverity.LOW

    def detect_anomalies(
        self,
        records: List[Dict],
        sort_by_time: bool = True,
    ) -> List[AnomalyRecord]:
        """Detect anomalies across a sequence of records.
        
        Each record should have at minimum:
        - zone_id (or PULocationID)
        - timestamp (or pickup_hour)
        - actual (or ride_count)
        Optional:
        - predicted (or predicted_rides)
        - lower_bound
        - upper_bound
        """
        if not records:
            return []

        df = pd.DataFrame(records)
        # Normalize column names
        if "PULocationID" in df.columns and "zone_id" not in df.columns:
            df["zone_id"] = df["PULocationID"]
        if "pickup_hour" in df.columns and "timestamp" not in df.columns:
            df["timestamp"] = df["pickup_hour"]
        if "ride_count" in df.columns and "actual" not in df.columns:
            df["actual"] = df["ride_count"]
        if "predicted_rides" in df.columns and "predicted" not in df.columns:
            df["predicted"] = df["predicted_rides"]

        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
        df["zone_id"] = df["zone_id"].astype(int)

        if sort_by_time:
            df = df.sort_values(["zone_id", "timestamp_dt"]).reset_index(drop=True)

        detected: List[AnomalyRecord] = []

        # Track persistent consecutive anomalous hours per zone
        consecutive_anomalies: Dict[int, int] = {}

        # Track duplicates
        seen_keys = set()

        for _, row in df.iterrows():
            zone_id = int(row["zone_id"])
            ts_dt = row["timestamp_dt"]
            ts_iso = ts_dt.strftime("%Y-%m-%dT%H:%M:%S")
            key = (zone_id, ts_iso)

            zone_info = self.zone_lookup.get(zone_id, {})
            zone_name = zone_info.get("Zone", f"Zone {zone_id}")
            borough = zone_info.get("Borough", "Unknown")

            # 1. Data Quality Check: Duplicate timestamp
            if key in seen_keys:
                detected.append(
                    AnomalyRecord(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        borough=borough,
                        timestamp=ts_iso,
                        actual_demand=float(row["actual"]) if pd.notnull(row.get("actual")) else None,
                        anomaly_type=AnomalyType.DATA_QUALITY,
                        severity=AnomalySeverity.HIGH,
                        explanation=f"Data quality defect: duplicate observation for zone {zone_id} at {ts_iso}.",
                        is_persistent=False,
                        consecutive_count=1,
                    )
                )
                continue
            seen_keys.add(key)

            raw_val = row.get("actual")

            # 2. Data Quality Check: Missing / NaN / Inf
            if raw_val is None or pd.isna(raw_val) or np.isinf(raw_val):
                detected.append(
                    AnomalyRecord(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        borough=borough,
                        timestamp=ts_iso,
                        actual_demand=None,
                        anomaly_type=AnomalyType.DATA_QUALITY,
                        severity=AnomalySeverity.HIGH,
                        explanation=f"Data quality defect: missing or invalid demand observation at {ts_iso}.",
                        is_persistent=False,
                        consecutive_count=1,
                    )
                )
                consecutive_anomalies[zone_id] = 0
                continue

            actual = float(raw_val)

            # 3. Data Quality Check: Negative or physically implausible demand
            if actual < 0:
                detected.append(
                    AnomalyRecord(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        borough=borough,
                        timestamp=ts_iso,
                        actual_demand=actual,
                        anomaly_type=AnomalyType.DATA_QUALITY,
                        severity=AnomalySeverity.CRITICAL,
                        explanation=f"Data quality defect: negative demand count ({actual:.0f} rides) is physically impossible.",
                        is_persistent=False,
                        consecutive_count=1,
                    )
                )
                consecutive_anomalies[zone_id] = 0
                continue

            if actual > self.config.MAX_PLAUSIBLE_HOURLY_DEMAND:
                detected.append(
                    AnomalyRecord(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        borough=borough,
                        timestamp=ts_iso,
                        actual_demand=actual,
                        anomaly_type=AnomalyType.DATA_QUALITY,
                        severity=AnomalySeverity.CRITICAL,
                        explanation=f"Data quality defect: extreme demand ({actual:.0f} rides) exceeds plausible physical ceiling ({self.config.MAX_PLAUSIBLE_HOURLY_DEMAND:.0f}).",
                        is_persistent=False,
                        consecutive_count=1,
                    )
                )
                consecutive_anomalies[zone_id] = 0
                continue

            # Retrieve seasonal expectation
            dow = ts_dt.day_of_week
            hour = ts_dt.hour
            med, mad = self._get_baseline_stats(zone_id, dow, hour)

            z = self.calculate_robust_z_score(actual, med, mad)
            dev = actual - med
            dev_pct = (dev / med * 100.0) if med > 0 else (100.0 if actual > 0 else 0.0)

            # Check forecast model prediction if provided
            predicted = float(row["predicted"]) if "predicted" in row and pd.notnull(row["predicted"]) else None
            lower_b = float(row["lower_bound"]) if "lower_bound" in row and pd.notnull(row["lower_bound"]) else None
            upper_b = float(row["upper_bound"]) if "upper_bound" in row and pd.notnull(row["upper_bound"]) else None

            # Detect demand spike
            is_spike = (
                z >= self.config.Z_THRESHOLD_LOW
                and dev >= self.config.MIN_ABSOLUTE_DEVIATION
            )

            # Detect demand drop (with low-volume safeguard)
            is_drop = (
                z <= -self.config.Z_THRESHOLD_LOW
                and dev <= -self.config.MIN_ABSOLUTE_DEVIATION
                and med >= self.config.MIN_BASELINE_VOLUME
            )

            # Detect unexpected forecast deviation
            is_forecast_dev = False
            forecast_reason = ""
            if predicted is not None:
                f_err = actual - predicted
                abs_f_err = abs(f_err)
                if lower_b is not None and upper_b is not None:
                    if (actual < lower_b - 5.0 or actual > upper_b + 5.0) and abs_f_err >= self.config.FORECAST_MIN_ABSOLUTE_DEVIATION:
                        is_forecast_dev = True
                        forecast_reason = f"Actual ({actual:.0f}) breached 80% conformal prediction interval [{lower_b:.0f}, {upper_b:.0f}] by {abs_f_err:.0f} rides."
                elif abs_f_err >= self.config.FORECAST_MIN_ABSOLUTE_DEVIATION:
                    rel_err = abs_f_err / max(predicted, 10.0) * 100.0
                    if rel_err >= self.config.FORECAST_RELATIVE_THRESHOLD_PCT:
                        is_forecast_dev = True
                        forecast_reason = f"Forecast error of {f_err:+.0f} rides ({rel_err:.0f}% deviation from predicted {predicted:.0f})."

            is_anomalous = is_spike or is_drop or is_forecast_dev

            if is_anomalous:
                consecutive_anomalies[zone_id] = consecutive_anomalies.get(zone_id, 0) + 1
            else:
                consecutive_anomalies[zone_id] = 0

            consec_count = consecutive_anomalies.get(zone_id, 0)
            is_persistent = consec_count >= self.config.PERSISTENT_WINDOW_HOURS

            if is_anomalous or is_persistent:
                # Determine classification
                if is_persistent and consec_count >= self.config.PERSISTENT_WINDOW_HOURS:
                    atype = AnomalyType.PERSISTENT_ANOMALY
                    explanation = f"Persistent demand anomaly: zone has exhibited unusual demand for {consec_count} consecutive hours. Current demand: {actual:.0f} rides (expected: {med:.0f})."
                elif is_spike:
                    atype = AnomalyType.DEMAND_SPIKE
                    explanation = f"Demand spike: observed {actual:.0f} rides, {dev_pct:+.0f}% above seasonal normal of {med:.0f} (robust z-score {z:+.2f})."
                elif is_drop:
                    atype = AnomalyType.DEMAND_DROP
                    explanation = f"Demand drop: observed {actual:.0f} rides, {dev_pct:+.0f}% below seasonal normal of {med:.0f} (robust z-score {z:+.2f})."
                else:
                    atype = AnomalyType.FORECAST_DEVIATION
                    explanation = f"Forecast deviation: {forecast_reason}"

                sev = self.determine_severity(
                    z_score=z,
                    is_data_quality=False,
                    is_persistent=is_persistent,
                    consecutive_count=consec_count,
                )

                detected.append(
                    AnomalyRecord(
                        zone_id=zone_id,
                        zone_name=zone_name,
                        borough=borough,
                        timestamp=ts_iso,
                        actual_demand=round(actual, 1),
                        baseline_demand=round(med, 1),
                        predicted_demand=round(predicted, 1) if predicted is not None else None,
                        lower_bound=round(lower_b, 1) if lower_b is not None else None,
                        upper_bound=round(upper_b, 1) if upper_b is not None else None,
                        deviation=round(dev, 1),
                        deviation_pct=round(dev_pct, 1),
                        z_score=round(z, 2),
                        anomaly_type=atype,
                        severity=sev,
                        explanation=explanation,
                        is_persistent=is_persistent,
                        consecutive_count=consec_count,
                    )
                )

        return detected
