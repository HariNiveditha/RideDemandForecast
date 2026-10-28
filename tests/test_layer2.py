"""Unit and integration tests for Layer 2 Demand Anomaly Detection and Error Analysis."""

import numpy as np
import pandas as pd
import pytest

from layer2.anomaly_detector import DemandAnomalyDetector
from layer2.config import Layer2Config
from layer2.error_analyzer import (
    compute_forecast_metrics,
    evaluate_daily_timeseries,
    evaluate_zone_breakdowns,
    generate_bias_interpretation,
)
from layer2.schemas import AnomalySeverity, AnomalyType
from layer2.service import Layer2Service


@pytest.fixture
def test_config():
    return Layer2Config()


@pytest.fixture
def fitted_detector(test_config):
    detector = DemandAnomalyDetector(test_config)
    # Synthetic training baseline: Zone 10 has median 50 rides, MAD 5 rides for Sunday 14:00 (dow=6, hour=14)
    # Zone 20 has quiet traffic: median 2 rides, MAD 1 ride
    history_records = []
    for dow in range(7):
        for hour in range(24):
            # Zone 10: median 50 on weekends 12-20 with spread +/- 10
            base = 50.0 if (dow in (5, 6) and 12 <= hour <= 20) else 30.0
            for i in range(-5, 6):  # 11 points: -5, -4, ..., 0, ..., 5 (median offset is 0.0)
                history_records.append({
                    "PULocationID": 10,
                    "dayofweek": dow,
                    "hour_of_day": hour,
                    "ride_count": base + (i * 2),  # offsets -10 to +10, MAD is 6.0
                })
            # Zone 20 (quiet zone)
            for i in range(11):
                history_records.append({
                    "PULocationID": 20,
                    "dayofweek": dow,
                    "hour_of_day": hour,
                    "ride_count": 2.0 if i < 9 else 3.0,
                })
    df_hist = pd.DataFrame(history_records)
    detector.fit_seasonal_profile(df_hist)
    detector.set_zone_lookup({
        10: {"Zone": "Times Square", "Borough": "Manhattan"},
        20: {"Zone": "City Island", "Borough": "Bronx"},
    })
    return detector


def test_seasonal_profile_calculation(fitted_detector):
    """Verify median and MAD profile extraction."""
    med, mad = fitted_detector._get_baseline_stats(10, dow=6, hour=14)
    assert med == 50.0
    assert mad >= 1.0

    z = fitted_detector.calculate_robust_z_score(actual=70.0, median=50.0, mad=5.0)
    # scale = max(1.4826 * 5.0, 1.0) = 7.413; z = 20.0 / 7.413 = 2.697
    assert 2.6 < z < 2.8


def test_demand_spike_detection(fitted_detector):
    """Verify demand spike detection and severity scoring."""
    records = [
        # Normal observation
        {
            "zone_id": 10,
            "timestamp": "2021-06-06T14:00:00",  # Sunday 14:00
            "actual": 52.0,
        },
        # Medium spike: 85 rides (median=50, dev=+35)
        {
            "zone_id": 10,
            "timestamp": "2021-06-13T14:00:00",
            "actual": 85.0,
        },
        # Critical spike: 150 rides (dev=+100, z > 7)
        {
            "zone_id": 10,
            "timestamp": "2021-06-20T14:00:00",
            "actual": 150.0,
        },
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    # The normal observation (52 rides) should not be flagged
    assert len(anomalies) == 2

    # Medium spike
    a1 = anomalies[0]
    assert a1.anomaly_type == AnomalyType.DEMAND_SPIKE
    assert a1.severity in (AnomalySeverity.MEDIUM, AnomalySeverity.HIGH)
    assert a1.deviation == 35.0
    assert "Demand spike" in a1.explanation

    # Critical spike
    a2 = anomalies[1]
    assert a2.anomaly_type == AnomalyType.DEMAND_SPIKE
    assert a2.severity == AnomalySeverity.CRITICAL
    assert a2.deviation == 100.0


def test_demand_drop_detection(fitted_detector):
    """Verify demand drop detection."""
    records = [
        # Severe drop: 15 rides when normal is 50 (dev = -35)
        {
            "zone_id": 10,
            "timestamp": "2021-06-06T14:00:00",
            "actual": 15.0,
        }
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.anomaly_type == AnomalyType.DEMAND_DROP
    assert a.deviation == -35.0
    assert a.z_score < -3.0
    assert "Demand drop" in a.explanation


def test_low_volume_safeguard(fitted_detector):
    """Verify quiet zones with small ride numbers do not trigger false spikes or drops."""
    records = [
        # Quiet zone 20 has median 2 rides; dropping to 0 or 1 is normal variation
        {"zone_id": 20, "timestamp": "2021-06-06T14:00:00", "actual": 0.0},
        {"zone_id": 20, "timestamp": "2021-06-06T15:00:00", "actual": 5.0},
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    # Neither should be flagged because deviations (|0 - 2| = 2, |5 - 2| = 3) are below MIN_ABSOLUTE_DEVIATION (12.0)
    assert len(anomalies) == 0


def test_forecast_deviation_detection(fitted_detector):
    """Verify unexpected forecast deviation when predicted vs actual differs materially."""
    records = [
        # Baseline is normal, but model predicted 100 while actual was 50 (error = -50, outside bounds)
        {
            "zone_id": 10,
            "timestamp": "2021-06-06T14:00:00",
            "actual": 50.0,
            "predicted": 100.0,
            "lower_bound": 85.0,
            "upper_bound": 115.0,
        }
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    assert len(anomalies) == 1
    a = anomalies[0]
    assert a.anomaly_type == AnomalyType.FORECAST_DEVIATION
    assert "conformal prediction interval" in a.explanation or "Forecast error" in a.explanation


def test_persistent_anomaly_detection(fitted_detector):
    """Verify 3+ consecutive anomalous hours triggers persistent anomaly flag and critical escalation."""
    records = [
        {"zone_id": 10, "timestamp": "2021-06-06T10:00:00", "actual": 90.0},  # Hour 1 anomalous
        {"zone_id": 10, "timestamp": "2021-06-06T11:00:00", "actual": 92.0},  # Hour 2 anomalous
        {"zone_id": 10, "timestamp": "2021-06-06T12:00:00", "actual": 95.0},  # Hour 3 anomalous -> Persistent!
        {"zone_id": 10, "timestamp": "2021-06-06T13:00:00", "actual": 98.0},  # Hour 4 anomalous -> Persistent + Critical!
        {"zone_id": 10, "timestamp": "2021-06-06T14:00:00", "actual": 50.0},  # Normal -> counter reset
        {"zone_id": 10, "timestamp": "2021-06-06T15:00:00", "actual": 90.0},  # Hour 1 of new streak
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    assert len(anomalies) == 5  # 4 from first streak + 1 from last hour

    # Hour 3 (3rd consecutive) should be flagged as persistent
    a3 = anomalies[2]
    assert a3.is_persistent is True
    assert a3.consecutive_count == 3
    assert a3.anomaly_type == AnomalyType.PERSISTENT_ANOMALY

    # Hour 4 (4th consecutive) should be critical
    a4 = anomalies[3]
    assert a4.is_persistent is True
    assert a4.consecutive_count == 4
    assert a4.severity == AnomalySeverity.CRITICAL

    # Hour 6 after reset: consecutive count should be 1
    a5 = anomalies[4]
    assert a5.is_persistent is False
    assert a5.consecutive_count == 1


def test_data_quality_anomalies(fitted_detector):
    """Verify negative, duplicate, missing, and implausible records are classified as data quality defects."""
    records = [
        # Negative count
        {"zone_id": 10, "timestamp": "2021-06-06T10:00:00", "actual": -15.0},
        # Implausible extreme count
        {"zone_id": 10, "timestamp": "2021-06-06T11:00:00", "actual": 10000.0},
        # NaN / None
        {"zone_id": 10, "timestamp": "2021-06-06T12:00:00", "actual": None},
        # Duplicate timestamp
        {"zone_id": 10, "timestamp": "2021-06-06T13:00:00", "actual": 50.0},
        {"zone_id": 10, "timestamp": "2021-06-06T13:00:00", "actual": 55.0},
    ]
    anomalies = fitted_detector.detect_anomalies(records)
    assert len(anomalies) == 4  # Negative, extreme, null, duplicate

    for a in anomalies:
        assert a.anomaly_type == AnomalyType.DATA_QUALITY
        assert "Data quality defect" in a.explanation


def test_forecast_error_convention_and_metrics():
    """Verify strict error convention: error = actual - predicted.
    Positive mean bias indicates underprediction.
    Negative mean bias indicates overprediction.
    """
    # 1. Underprediction case: actual demand higher than predicted
    actuals_under = np.array([100.0, 120.0, 140.0])
    preds_under = np.array([80.0, 100.0, 110.0])
    res_under = compute_forecast_metrics(actuals_under, preds_under, horizon_minutes=60)
    assert res_under.mean_bias > 0  # +23.33 -> Positive signed bias
    assert "underprediction" in res_under.interpretation
    assert res_under.mae == pytest.approx(23.333, abs=0.01)

    # 2. Overprediction case: forecast higher than actual demand
    actuals_over = np.array([50.0, 60.0, 70.0])
    preds_over = np.array([70.0, 80.0, 95.0])
    res_over = compute_forecast_metrics(actuals_over, preds_over, horizon_minutes=60)
    assert res_over.mean_bias < 0  # -21.67 -> Negative signed bias
    assert "overprediction" in res_over.interpretation

    # 3. WAPE and interval coverage
    actuals = np.array([50.0, 100.0])
    preds = np.array([60.0, 80.0])
    # abs errors: 10 + 20 = 30. sum actuals: 150. WAPE = 30 / 150 * 100 = 20.0%
    lb = np.array([45.0, 90.0])
    ub = np.array([75.0, 110.0])
    # 50 in [45, 75] -> yes. 100 in [90, 110] -> yes. Coverage = 100%
    res = compute_forecast_metrics(actuals, preds, lower_bound=lb, upper_bound=ub, horizon_minutes=60)
    assert res.wape_pct == 20.0
    assert res.empirical_coverage_pct == 100.0
    assert res.mean_interval_width == pytest.approx(25.0, abs=0.01)


def test_layer2_service_end_to_end():
    """Verify Layer 2 service operations, filters, and summaries."""
    service = Layer2Service.get_default_service()
    assert service._is_initialized is True

    # 1. Anomalies endpoint simulation
    anomalies = service.get_anomalies(limit=25)
    assert len(anomalies) <= 25
    for a in anomalies:
        assert a.zone_id > 0
        assert a.timestamp is not None
        assert a.explanation

    # 2. Filter by anomaly type
    spikes = service.get_anomalies(anomaly_type=AnomalyType.DEMAND_SPIKE, limit=10)
    for s in spikes:
        assert s.anomaly_type == AnomalyType.DEMAND_SPIKE

    # 3. Summary endpoint
    summary = service.get_anomaly_summary()
    assert summary.total_anomalies >= 0
    assert "demand_spike" in summary.by_type
    assert "demand_drop" in summary.by_type
    assert "persistent_anomaly" in summary.by_type

    # 4. Forecast performance endpoint
    perf = service.get_forecast_performance()
    assert len(perf.horizons) == 3
    horizons = [h.horizon_minutes for h in perf.horizons]
    assert horizons == [60, 180, 360]
    for h in perf.horizons:
        assert h.mae > 0
        assert h.rmse > 0
        assert h.sample_count > 0

    # 5. Zone errors
    zone_errors = service.get_zone_errors(horizon_minutes=60, sort_by="mae", limit=5)
    assert len(zone_errors) <= 5
    # Verify sorted descending by MAE
    for i in range(len(zone_errors) - 1):
        assert zone_errors[i].mae >= zone_errors[i + 1].mae

    # 6. Timeseries errors
    ts = service.get_timeseries_errors(horizon_minutes=60)
    assert len(ts) == 30  # 30 days of June
    assert ts[0].date.startswith("2021-06-")
