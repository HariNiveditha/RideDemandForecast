"""Tests for prediction intervals, coverage computation, and non-negativity constraints."""

import numpy as np
import pytest

from forecasting.uncertainty import ConformalResidualCalibrator


def test_conformal_calibration_and_bounds():
    calibrator = ConformalResidualCalibrator(confidence_level=0.80)

    # Simulated validation data
    np.random.seed(42)
    y_true = np.random.poisson(lam=50.0, size=2000).astype(float)
    noise = np.random.normal(loc=0.0, scale=8.0, size=2000)
    y_pred = np.maximum(0.0, y_true + noise)

    metrics = calibrator.calibrate(y_true, y_pred, horizon_minutes=60)
    assert metrics["target_coverage_pct"] == 80.0
    assert 78.0 <= metrics["empirical_coverage_pct"] <= 82.0

    # Test bounds for single scalar
    lower, upper = calibrator.predict_bounds(y_pred=45.0, horizon_minutes=60)
    assert lower >= 0.0
    assert upper >= lower

    # Test non-negative clamping for low prediction
    low_lower, low_upper = calibrator.predict_bounds(y_pred=2.0, horizon_minutes=60)
    assert low_lower == 0.0
    assert low_upper >= 0.0

    # Test test-set evaluation
    test_metrics = calibrator.evaluate_test_set(y_true, y_pred, horizon_minutes=60)
    assert test_metrics["empirical_test_coverage_pct"] >= 78.0
