"""Configuration and thresholds for Layer 2 Anomaly Detection and Error Analysis."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layer2Config:
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / "ride_demand_weather_hourly.parquet"
    ZONE_LOOKUP_PATH: Path = BASE_DIR / "taxi_zone_lookup.csv"
    FORECAST_EVAL_PATH: Path = BASE_DIR / "forecasting" / "artifacts" / "multi_horizon_evaluation_results.json"
    LAYER2_CACHE_DIR: Path = BASE_DIR / "forecasting" / "artifacts" / "layer2_cache"

    # Robust statistics constants
    # 1.4826 makes MAD asymptotically consistent with standard deviation for normal distributions
    MAD_SCALE_FACTOR: float = 1.4826
    EPSILON_STD: float = 1.0

    # Z-score thresholds
    Z_THRESHOLD_LOW: float = 2.5
    Z_THRESHOLD_MEDIUM: float = 3.5
    Z_THRESHOLD_HIGH: float = 5.0
    Z_THRESHOLD_CRITICAL: float = 7.0

    # Volume safeguards: prevent flagging minor variations in low-traffic zones
    MIN_ABSOLUTE_DEVIATION: float = 12.0
    MIN_BASELINE_VOLUME: float = 5.0

    # Persistent anomaly trigger
    PERSISTENT_WINDOW_HOURS: int = 3

    # Forecast deviation thresholds
    FORECAST_RELATIVE_THRESHOLD_PCT: float = 50.0
    FORECAST_MIN_ABSOLUTE_DEVIATION: float = 15.0

    # Data quality boundaries
    MAX_PLAUSIBLE_HOURLY_DEMAND: float = 3000.0
    ALLOW_NEGATIVE_DEMAND: bool = False
