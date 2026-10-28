"""Data models and schemas for Layer 2 Anomaly Detection and Forecast Error Analysis."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnomalyType(str, Enum):
    DEMAND_SPIKE = "demand_spike"
    DEMAND_DROP = "demand_drop"
    FORECAST_DEVIATION = "forecast_deviation"
    PERSISTENT_ANOMALY = "persistent_anomaly"
    DATA_QUALITY = "data_quality"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyRecord(BaseModel):
    zone_id: int = Field(gt=0, description="NYC TLC Taxi Zone ID")
    zone_name: Optional[str] = Field(default=None, description="Human-readable zone name")
    borough: Optional[str] = Field(default=None, description="Borough name")
    timestamp: str = Field(description="ISO 8601 timestamp of anomalous interval")
    actual_demand: Optional[float] = Field(default=None, description="Observed ride count")
    baseline_demand: Optional[float] = Field(default=None, description="Expected seasonal median demand")
    predicted_demand: Optional[float] = Field(default=None, description="Layer 1 model prediction if available")
    lower_bound: Optional[float] = Field(default=None, description="Lower prediction interval bound")
    upper_bound: Optional[float] = Field(default=None, description="Upper prediction interval bound")
    deviation: Optional[float] = Field(default=None, description="Signed deviation (actual - expected)")
    deviation_pct: Optional[float] = Field(default=None, description="Percentage deviation from expected")
    z_score: Optional[float] = Field(default=None, description="Robust z-score based on MAD")
    anomaly_type: AnomalyType = Field(description="Categorical anomaly type")
    severity: AnomalySeverity = Field(description="Deterministic severity level")
    explanation: str = Field(description="Concise, human-readable reason for the flag")
    is_persistent: bool = Field(default=False, description="Whether this interval is part of a consecutive streak")
    consecutive_count: int = Field(default=1, description="Number of consecutive anomalous intervals")


class AnomalySummaryResponse(BaseModel):
    total_anomalies: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    top_zones: List[Dict[str, Any]]
    date_range: Dict[str, str]


class ForecastMetricSummary(BaseModel):
    horizon_minutes: int
    horizon_hours: int
    mae: float
    rmse: float
    mean_bias: float = Field(description="Signed bias: mean(actual - predicted). Positive=underprediction, Negative=overprediction")
    wape_pct: float
    sample_count: int
    empirical_coverage_pct: float
    mean_interval_width: float
    interpretation: str


class ForecastPerformanceResponse(BaseModel):
    horizons: List[ForecastMetricSummary]
    model_version: str
    evaluation_period: Dict[str, str]
    error_convention: str = "error = actual - predicted (positive = underprediction, negative = overprediction)"


class ZoneErrorRecord(BaseModel):
    zone_id: int
    zone_name: str
    borough: str
    mae: float
    rmse: float
    mean_bias: float
    wape_pct: float
    sample_count: int


class TimeSeriesErrorPoint(BaseModel):
    date: str
    horizon_minutes: int
    mae: float
    rmse: float
    mean_bias: float
    sample_count: int
