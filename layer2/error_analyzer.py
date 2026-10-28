"""Forecast error analysis and multi-horizon performance evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from layer2.schemas import (
    ForecastMetricSummary,
    TimeSeriesErrorPoint,
    ZoneErrorRecord,
)


def compute_forecast_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    lower_bound: Optional[np.ndarray | pd.Series] = None,
    upper_bound: Optional[np.ndarray | pd.Series] = None,
    horizon_minutes: int = 60,
) -> ForecastMetricSummary:
    """Compute core forecast evaluation metrics following the strict error convention:
    error = actual - predicted.
    
    Positive mean bias indicates underprediction (actual demand exceeded forecast).
    Negative mean bias indicates overprediction (forecast exceeded actual demand).
    """
    y_true = np.asarray(actual, dtype=float)
    y_pred = np.asarray(predicted, dtype=float)

    # Valid mask for finite non-null entries
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        return ForecastMetricSummary(
            horizon_minutes=horizon_minutes,
            horizon_hours=horizon_minutes // 60,
            mae=0.0,
            rmse=0.0,
            mean_bias=0.0,
            wape_pct=0.0,
            sample_count=0,
            empirical_coverage_pct=0.0,
            mean_interval_width=0.0,
            interpretation="No valid sample observations to evaluate.",
        )

    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    n_samples = int(len(y_t))

    errors = y_t - y_p  # Convention: actual - predicted
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mean_bias = float(np.mean(errors))

    sum_actual = float(np.sum(y_t))
    if sum_actual > 0:
        wape_pct = float(np.sum(abs_errors) / sum_actual * 100.0)
    else:
        wape_pct = 0.0

    # Uncertainty coverage and interval width
    if lower_bound is not None and upper_bound is not None:
        lb = np.asarray(lower_bound, dtype=float)[valid_mask]
        ub = np.asarray(upper_bound, dtype=float)[valid_mask]
        covered = (y_t >= lb) & (y_t <= ub)
        empirical_coverage_pct = float(np.mean(covered) * 100.0)
        mean_interval_width = float(np.mean(ub - lb))
    else:
        empirical_coverage_pct = 0.0
        mean_interval_width = 0.0

    interpretation = generate_bias_interpretation(mean_bias)

    return ForecastMetricSummary(
        horizon_minutes=horizon_minutes,
        horizon_hours=horizon_minutes // 60,
        mae=round(mae, 3),
        rmse=round(rmse, 3),
        mean_bias=round(mean_bias, 3),
        wape_pct=round(wape_pct, 2),
        sample_count=n_samples,
        empirical_coverage_pct=round(empirical_coverage_pct, 2),
        mean_interval_width=round(mean_interval_width, 2),
        interpretation=interpretation,
    )


def generate_bias_interpretation(mean_bias: float) -> str:
    """Generate clear human-readable interpretation of signed mean bias."""
    if abs(mean_bias) < 0.2:
        return f"Well-calibrated: near zero bias ({mean_bias:+.2f} rides/hr)."
    elif mean_bias > 0:
        return f"Systematic underprediction: actual demand averages {abs(mean_bias):.2f} rides higher than predicted."
    else:
        return f"Systematic overprediction: forecast averages {abs(mean_bias):.2f} rides higher than actual demand."


def evaluate_daily_timeseries(
    df: pd.DataFrame,
    horizon_minutes: int = 60,
) -> List[TimeSeriesErrorPoint]:
    """Compute daily error metrics over time to identify temporal drift or anomalous days."""
    required = {"date", "actual", "predicted"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required}")

    points: List[TimeSeriesErrorPoint] = []
    for day, group in df.groupby("date"):
        y_t = group["actual"].to_numpy(dtype=float)
        y_p = group["predicted"].to_numpy(dtype=float)
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        if not np.any(valid):
            continue

        errors = y_t[valid] - y_p[valid]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))

        points.append(
            TimeSeriesErrorPoint(
                date=str(day),
                horizon_minutes=horizon_minutes,
                mae=round(mae, 3),
                rmse=round(rmse, 3),
                mean_bias=round(bias, 3),
                sample_count=int(np.sum(valid)),
            )
        )

    points.sort(key=lambda p: p.date)
    return points


def evaluate_zone_breakdowns(
    df: pd.DataFrame,
    zone_lookup: Optional[Dict[int, Dict[str, str]]] = None,
) -> List[ZoneErrorRecord]:
    """Compute per-zone forecast error metrics including signed bias, MAE, and WAPE."""
    required = {"PULocationID", "actual", "predicted"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required}")

    records: List[ZoneErrorRecord] = []
    zone_lookup = zone_lookup or {}

    for zone_id, group in df.groupby("PULocationID"):
        y_t = group["actual"].to_numpy(dtype=float)
        y_p = group["predicted"].to_numpy(dtype=float)
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        if not np.any(valid):
            continue

        errors = y_t[valid] - y_p[valid]
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        bias = float(np.mean(errors))
        sum_act = float(np.sum(y_t[valid]))
        wape = float(np.sum(np.abs(errors)) / sum_act * 100.0) if sum_act > 0 else 0.0

        zone_meta = zone_lookup.get(int(zone_id), {"Zone": f"Zone {zone_id}", "Borough": "Unknown"})

        records.append(
            ZoneErrorRecord(
                zone_id=int(zone_id),
                zone_name=zone_meta.get("Zone", f"Zone {zone_id}"),
                borough=zone_meta.get("Borough", "Unknown"),
                mae=round(mae, 3),
                rmse=round(rmse, 3),
                mean_bias=round(bias, 3),
                wape_pct=round(wape, 2),
                sample_count=int(np.sum(valid)),
            )
        )

    return records
