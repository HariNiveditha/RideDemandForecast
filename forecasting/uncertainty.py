"""Uncertainty quantification and calibrated empirical prediction intervals."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting.config import ARTIFACTS_DIR, DEFAULT_HORIZONS_MINUTES, DEFAULT_INTERVAL_CONFIDENCE

logger = logging.getLogger(__name__)


class ConformalResidualCalibrator:
    """Split conformal residual calibrator for multi-horizon point forecasts.
    
    Calibration procedure:
      1. On a chronological validation set (e.g. May 16 - May 31), generate point forecasts.
      2. Compute signed residuals: e_i = y_i - y_hat_i.
      3. For a desired nominal confidence level (e.g. 80%), compute empirical quantiles
         q_low = quantile(residuals, alpha / 2) and q_high = quantile(residuals, 1 - alpha / 2).
      4. For any future point prediction y_hat:
         lower_bound = max(0.0, y_hat + q_low)
         upper_bound = max(lower_bound, y_hat + q_high)
    
    Assumptions & Limitations:
      - Exchangeability: The distribution of residuals in future periods is assumed similar
        to the calibration window.
      - Non-negative demand: Lower bound is clamped at 0.0 since negative taxi ride counts
        are physically impossible.
      - Unmodeled structural shifts (e.g. unprecedented blizzards, transit strikes, holidays)
        may temporarily reduce empirical coverage.
    """

    def __init__(self, confidence_level: float = DEFAULT_INTERVAL_CONFIDENCE):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        # Quantile offsets per horizon: {horizon_minutes: (q_low, q_high)}
        self.calibration_table: Dict[int, Tuple[float, float]] = {}
        self.validation_metrics: Dict[int, Dict[str, float]] = {}

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_minutes: int,
    ) -> Dict[str, float]:
        """Calibrate quantiles on validation set for a given horizon."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        residuals = y_true - y_pred
        q_low = float(np.percentile(residuals, (self.alpha / 2.0) * 100.0))
        q_high = float(np.percentile(residuals, (1.0 - (self.alpha / 2.0)) * 100.0))

        self.calibration_table[horizon_minutes] = (round(q_low, 3), round(q_high, 3))

        # Check in-sample validation coverage
        lower = np.maximum(0.0, y_pred + q_low)
        upper = np.maximum(lower, y_pred + q_high)
        cov = float(np.mean((y_true >= lower) & (y_true <= upper)) * 100.0)
        mean_width = float(np.mean(upper - lower))

        metrics = {
            "target_coverage_pct": round(self.confidence_level * 100.0, 1),
            "empirical_coverage_pct": round(cov, 2),
            "mean_interval_width": round(mean_width, 2),
            "q_low": round(q_low, 3),
            "q_high": round(q_high, 3),
            "sample_size": len(y_true),
        }
        self.validation_metrics[horizon_minutes] = metrics
        return metrics

    def predict_bounds(
        self,
        y_pred: np.ndarray | float,
        horizon_minutes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals [lower_bound, upper_bound] for point forecasts."""
        if horizon_minutes not in self.calibration_table:
            # Fallback default quantile if horizon was not calibrated
            q_low, q_high = (-15.0, 15.0)
        else:
            q_low, q_high = self.calibration_table[horizon_minutes]

        is_scalar = np.isscalar(y_pred)
        arr = np.asarray(y_pred, dtype=float)

        lower = np.maximum(0.0, arr + q_low)
        upper = np.maximum(lower, arr + q_high)

        if is_scalar:
            return float(lower.item()), float(upper.item())
        return lower, upper

    def evaluate_test_set(
        self,
        y_test: np.ndarray,
        y_pred_test: np.ndarray,
        horizon_minutes: int,
    ) -> Dict[str, float]:
        """Evaluate empirical coverage and interval width on held-out test split."""
        lower, upper = self.predict_bounds(y_pred_test, horizon_minutes)
        y_test = np.asarray(y_test, dtype=float)

        covered = (y_test >= lower) & (y_test <= upper)
        cov = float(np.mean(covered) * 100.0)
        mean_width = float(np.mean(upper - lower))

        return {
            "target_confidence_pct": round(self.confidence_level * 100.0, 1),
            "empirical_test_coverage_pct": round(cov, 2),
            "mean_interval_width": round(mean_width, 2),
            "q_low": self.calibration_table.get(horizon_minutes, (0.0, 0.0))[0],
            "q_high": self.calibration_table.get(horizon_minutes, (0.0, 0.0))[1],
            "test_sample_size": len(y_test),
        }

    def save(self, directory: Optional[Path] = None) -> Path:
        """Save calibration tables to json."""
        save_dir = directory or ARTIFACTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "uncertainty_calibration.json"

        data = {
            "confidence_level": self.confidence_level,
            "calibration_table": {str(k): v for k, v in self.calibration_table.items()},
            "validation_metrics": {str(k): v for k, v in self.validation_metrics.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "ConformalResidualCalibrator":
        """Load calibration tables from disk."""
        load_dir = directory or ARTIFACTS_DIR
        path = load_dir / "uncertainty_calibration.json"
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(confidence_level=data.get("confidence_level", DEFAULT_INTERVAL_CONFIDENCE))
        instance.calibration_table = {int(k): tuple(v) for k, v in data.get("calibration_table", {}).items()}
        instance.validation_metrics = {int(k): v for k, v in data.get("validation_metrics", {}).items()}
        return instance
