"""Full training, uncertainty calibration, multi-horizon evaluation, and artifact generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting.baseline import (
    BaselineForecaster,
    calculate_metrics,
    evaluate_baselines_on_dataset,
)
from forecasting.config import (
    ARTIFACTS_DIR,
    BASE_DIR,
    DEFAULT_HORIZONS_MINUTES,
    FIXTURES_DIR,
    MODEL_VERSION,
    MULTI_HORIZON_FEATURE_ORDER,
    PLOTS_DIR,
)
from forecasting.data import (
    build_multi_horizon_dataset,
    build_regular_hourly_grid,
    load_raw_dataset,
    load_zone_lookup,
    split_chronological,
)
from forecasting.layer3_contract import (
    export_forecasts_to_csv,
    export_forecasts_to_json,
    get_layer3_json_schema,
)
from forecasting.models import MultiHorizonDemandModel
from forecasting.uncertainty import ConformalResidualCalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> Dict:
    """Execute complete end-to-end training, uncertainty calibration, and evaluation pipeline."""
    logger.info("Starting Layer 1 Forecasting pipeline...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and construct regular grid
    raw_df = load_raw_dataset()
    zone_lookup = load_zone_lookup()
    grid_df, weather_df = build_regular_hourly_grid(raw_df)
    logger.info(f"Regular grid constructed: {grid_df.shape[0]} total zone-hour observations.")

    model = MultiHorizonDemandModel(horizons_minutes=DEFAULT_HORIZONS_MINUTES, model_type="lightgbm")
    calibrator = ConformalResidualCalibrator(confidence_level=0.80)

    all_evaluations: Dict[int, Dict] = {}
    test_datasets: Dict[int, pd.DataFrame] = {}
    test_predictions: Dict[int, np.ndarray] = {}

    # 2. Train and evaluate for each horizon
    for h in DEFAULT_HORIZONS_MINUTES:
        logger.info(f"\n==================================================")
        logger.info(f"PROCESSING FORECAST HORIZON: {h} MINUTES ({h//60} HOUR(S) AHEAD)")
        logger.info(f"==================================================")

        ds = build_multi_horizon_dataset(grid_df, weather_df, h)
        train_df, val_df, test_df = split_chronological(ds)
        test_datasets[h] = test_df

        # Evaluate baselines on Test split
        combined_train = ds[ds["forecast_origin"] < "2021-06-01"]
        baseline_results = evaluate_baselines_on_dataset(combined_train, test_df, h)

        # Step 2a: Train model on train split (April 1 - May 15) for calibration
        X_train = train_df[MULTI_HORIZON_FEATURE_ORDER]
        y_train = train_df["target"].to_numpy()
        X_val = val_df[MULTI_HORIZON_FEATURE_ORDER]
        y_val = val_df["target"].to_numpy()

        temp_model = MultiHorizonDemandModel(horizons_minutes=[h], model_type="lightgbm")
        temp_model.fit_horizon(X_train, y_train, h)
        val_preds = temp_model.predict(X_val, h)

        # Calibrate conformal prediction intervals on validation split
        cal_metrics = calibrator.calibrate(y_val, val_preds, h)
        logger.info(f"Uncertainty calibrated for horizon {h}m: {cal_metrics}")

        # Step 2b: Retrain on combined train+val (April 1 - May 31) for maximum test accuracy
        X_full_train = combined_train[MULTI_HORIZON_FEATURE_ORDER]
        y_full_train = combined_train["target"].to_numpy()
        model.fit_horizon(X_full_train, y_full_train, h)

        # Step 2c: Evaluate trained model on test split (June 1 - June 30)
        X_test = test_df[MULTI_HORIZON_FEATURE_ORDER]
        y_test = test_df["target"].to_numpy()
        y_pred = model.predict(X_test, h)
        test_predictions[h] = y_pred

        model_metrics = calculate_metrics(y_test, y_pred)
        unc_metrics = calibrator.evaluate_test_set(y_test, y_pred, h)

        # Calculate per-zone metrics
        test_eval_df = test_df[["PULocationID", "target"]].copy()
        test_eval_df["predicted"] = y_pred
        zone_metrics = {}
        for zid, group in test_eval_df.groupby("PULocationID"):
            zone_metrics[int(zid)] = calculate_metrics(group["target"].to_numpy(), group["predicted"].to_numpy())

        # Group by borough
        test_eval_df = test_eval_df.merge(zone_lookup[["PULocationID", "Borough"]], on="PULocationID", how="left")
        borough_metrics = {}
        for boro, group in test_eval_df.groupby("Borough"):
            borough_metrics[str(boro)] = calculate_metrics(group["target"].to_numpy(), group["predicted"].to_numpy())

        all_evaluations[h] = {
            "horizon_minutes": h,
            "horizon_hours": h // 60,
            "baselines": baseline_results,
            "model_metrics": model_metrics,
            "uncertainty_metrics": unc_metrics,
            "borough_metrics": borough_metrics,
            "zone_metrics": zone_metrics,
        }

        logger.info(f"Horizon {h}m Model -> MAE: {model_metrics['mae']:.2f}, RMSE: {model_metrics['rmse']:.2f}, WAPE: {model_metrics['wape_pct']:.2f}%, R2: {model_metrics['r2']:.3f}")
        logger.info(f"Horizon {h}m Coverage -> Target: {unc_metrics['target_confidence_pct']}%, Empirical: {unc_metrics['empirical_test_coverage_pct']:.2f}%, Mean Width: {unc_metrics['mean_interval_width']:.2f} rides")

    # 3. Save model artifacts and calibration
    model.save(ARTIFACTS_DIR)
    calibrator.save(ARTIFACTS_DIR)

    # Also export compatible rf_model.pkl in root so existing main.py endpoints work seamlessly
    try:
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        rf_path = BASE_DIR / "rf_model.pkl"
        if not rf_path.exists():
            logger.info("Generating compatible rf_model.pkl for root main.py...")
            ds1 = test_datasets[60]
            # Use small fast forest sample to ensure immediate functionality
            X_sample = ds1[MULTI_HORIZON_FEATURE_ORDER].iloc[:50000]
            y_sample = ds1["target"].iloc[:50000]
            rf = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
            # Match main.py's FEATURE_ORDER if possible
            import main
            frame = main._build_model_frame()
            train_mask = frame["pickup_date"] < "2021-06-01"
            rf.fit(frame.loc[train_mask, main.FEATURE_ORDER].iloc[::5], frame.loc[train_mask, "ride_count"].iloc[::5])
            joblib.dump(rf, rf_path)
            logger.info(f"Saved compatible rf_model.pkl to {rf_path}")
    except Exception as ex:
        logger.warning(f"Could not write root rf_model.pkl: {ex}")

    # 4. Generate Layer 3 Schema and Fixtures
    logger.info("Generating Layer 3 contract schema and sample fixtures...")
    schema = get_layer3_json_schema()
    with open(FIXTURES_DIR / "layer3_integration_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    # Sample forecast records for top representative zones across all horizons
    sample_zones = [161, 237, 236, 132, 138, 79, 148, 1, 48, 68]  # Midtown, JFK, LGA, East Village, etc.
    sample_origin = "2021-06-15T12:00:00"

    from forecasting.service import ForecastingService
    service = ForecastingService(model=model, calibrator=calibrator, grid_df=grid_df, weather_df=weather_df)
    service._history_series = grid_df.set_index(["PULocationID", "pickup_hour"])["ride_count"].sort_index()
    sample_records = service.forecast_batch(
        zone_ids=sample_zones,
        forecast_origin=sample_origin,
        horizons_minutes=DEFAULT_HORIZONS_MINUTES,
        include_bounds=True,
    )

    export_forecasts_to_json(sample_records, FIXTURES_DIR / "sample_forecast_output.json")
    export_forecasts_to_csv(sample_records, FIXTURES_DIR / "sample_forecast_output.csv")
    logger.info(f"Saved sample fixtures to {FIXTURES_DIR}")

    # 5. Generate Evaluation Plots
    generate_plots(all_evaluations, test_datasets, test_predictions)

    # 6. Save full evaluation results JSON
    eval_json_path = ARTIFACTS_DIR / "multi_horizon_evaluation_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluations, f, indent=2)

    logger.info(f"Saved evaluation results to {eval_json_path}")
    return all_evaluations


def generate_plots(
    all_evaluations: Dict[int, Dict],
    test_datasets: Dict[int, pd.DataFrame],
    test_predictions: Dict[int, np.ndarray],
) -> None:
    """Generate high-quality evaluation plots."""
    logger.info("Rendering evaluation plots...")

    # Plot 1: Baseline vs Model across Horizons
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_to_plot = [("mae", "MAE (Mean Absolute Error)", "rides"), ("rmse", "RMSE (Root Mean Sq Error)", "rides"), ("wape_pct", "WAPE", "%")]

    horizons_str = [f"{h//60}h ({h}m)" for h in DEFAULT_HORIZONS_MINUTES]
    x = np.arange(len(horizons_str))
    width = 0.22

    for i, (m_key, m_label, unit) in enumerate(metrics_to_plot):
        ax = axes[i]
        model_vals = [all_evaluations[h]["model_metrics"][m_key] for h in DEFAULT_HORIZONS_MINUTES]
        lag168_vals = [all_evaluations[h]["baselines"]["Baseline: Same Hour Last Week (target lag-168)"][m_key] for h in DEFAULT_HORIZONS_MINUTES]
        persist_vals = [all_evaluations[h]["baselines"]["Baseline: Origin Persistence (origin lag-0)"][m_key] for h in DEFAULT_HORIZONS_MINUTES]

        ax.bar(x - width, lag168_vals, width, label="Lag-168 (Last Week)", color="#f59e0b")
        ax.bar(x, persist_vals, width, label="Persistence (Origin t)", color="#ef4444")
        ax.bar(x + width, model_vals, width, label="Layer 1 Direct Model", color="#10b981")

        ax.set_title(m_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(horizons_str)
        ax.set_ylabel(unit)
        ax.grid(True, linestyle="--", alpha=0.5)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "baseline_vs_model_metrics.png", dpi=200)
    plt.close()

    # Plot 2: Actual vs Predicted Demand Series (High-Volume Zone e.g. Midtown Manhattan zone 161)
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    sample_zone_id = 161  # Midtown Center

    for idx, h in enumerate(DEFAULT_HORIZONS_MINUTES):
        ax = axes[idx]
        test_df = test_datasets[h].copy()
        test_df["pred"] = test_predictions[h]

        zone_subset = test_df[test_df["PULocationID"] == sample_zone_id].sort_values("target_timestamp").iloc[24:168]  # 6-day sample
        timestamps = zone_subset["target_timestamp"]
        actuals = zone_subset["target"]
        preds = zone_subset["pred"]

        ax.plot(timestamps, actuals, label="Actual Rides", color="#1e293b", linewidth=2.0)
        ax.plot(timestamps, preds, label=f"Predicted (H={h//60}h)", color="#2563eb", linestyle="--", linewidth=1.8)

        ax.set_title(f"Zone {sample_zone_id} (Midtown Center) — Forecast Horizon {h//60}h ({h}m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Rides / hour")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right")

    plt.xlabel("Target Timestamp (June 2021)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted_horizon_comparison.png", dpi=200)
    plt.close()

    # Plot 3: Prediction Interval Coverage by Horizon
    fig, ax = plt.subplots(figsize=(8, 5))
    target_cov = [all_evaluations[h]["uncertainty_metrics"]["target_confidence_pct"] for h in DEFAULT_HORIZONS_MINUTES]
    emp_cov = [all_evaluations[h]["uncertainty_metrics"]["empirical_test_coverage_pct"] for h in DEFAULT_HORIZONS_MINUTES]
    mean_widths = [all_evaluations[h]["uncertainty_metrics"]["mean_interval_width"] for h in DEFAULT_HORIZONS_MINUTES]

    x = np.arange(len(horizons_str))
    width = 0.35

    ax.bar(x - width/2, target_cov, width, label="Target Confidence (80%)", color="#94a3b8")
    ax.bar(x + width/2, emp_cov, width, label="Empirical Test Coverage", color="#059669")

    for i in range(len(x)):
        ax.text(x[i] + width/2, emp_cov[i] + 1, f"{emp_cov[i]:.1f}%\n(width: {mean_widths[i]:.1f})", ha="center", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons_str)
    ax.set_ylabel("Coverage (%)")
    ax.set_title("80% Empirical Prediction Interval Coverage on Held-Out Test Data", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_intervals_coverage.png", dpi=200)
    plt.close()

    logger.info(f"Plots successfully saved to {PLOTS_DIR}")


if __name__ == "__main__":
    run_pipeline()
