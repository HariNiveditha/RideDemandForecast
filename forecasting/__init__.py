"""Layer 1: Reliable Zone-Level Multi-Horizon Demand Forecasting module."""

from forecasting.layer3_contract import ForecastRecord, Layer3Adapter
from forecasting.service import ForecastingService, forecast_batch, predict_zone

__version__ = "1.0.0"

__all__ = [
    "ForecastingService",
    "predict_zone",
    "forecast_batch",
    "ForecastRecord",
    "Layer3Adapter",
]
