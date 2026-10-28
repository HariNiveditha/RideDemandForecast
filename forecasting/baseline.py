"""Baseline forecasting models and evaluation metrics for ride demand forecasting."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, WAPE, and R2 on predictions.
    
    WAPE = sum(|y - y_hat|) / sum(y) * 100%, perfectly defined when zero demand intervals exist.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Clamping negative predictions to zero (physical constraint on ride counts)
    y_pred = np.maximum(y_pred, 0.0)

    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(sq_diff)))

    total_actual = float(np.sum(y_true))
    wape = float((np.sum(abs_diff) / total_actual) * 100.0) if total_actual > 0 else 0.0

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(sq_diff))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "wape_pct": round(wape, 2),
        "r2": round(r2, 4),
        "count": len(y_true),
        "total_actual_rides": round(total_actual, 1),
    }


class BaselineForecaster:
    """Collection of naive and statistical baseline forecasters for multi-horizon demand."""

    @staticmethod
    def predict_lag_target_168(df: pd.DataFrame) -> np.ndarray:
        """Predict using the demand from the exact same hour one week prior to the target."""
        return df["target_lag_168"].fillna(0.0).to_numpy()

    @staticmethod
    def predict_lag_target_24(df: pd.DataFrame) -> np.ndarray:
        """Predict using demand from the same hour one day prior to target."""
        return df["target_lag_24"].fillna(0.0).to_numpy()

    @staticmethod
    def predict_persistence_origin(df: pd.DataFrame) -> np.ndarray:
        """Persistence baseline: predict current demand at origin T for all future horizons."""
        return df["origin_lag_0"].fillna(0.0).to_numpy()

    @staticmethod
    def predict_rolling_mean_24(df: pd.DataFrame) -> np.ndarray:
        """Predict the 24-hour rolling average up to origin T."""
        return df["origin_rolling_mean_24"].fillna(0.0).to_numpy()


class HistoricalProfileBaseline:
    """Historical mean demand baseline by (zone, day_of_week, hour_of_day), fitted only on train split."""

    def __init__(self):
        self.profile_: Dict[Tuple[int, int, int], float] = {}
        self.global_mean_: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "HistoricalProfileBaseline":
        grouped = train_df.groupby(["PULocationID", "dayofweek", "hour_of_day"])["target"].mean()
        self.profile_ = grouped.to_dict()
        self.global_mean_ = float(train_df["target"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for loc, dow, hour in zip(df["PULocationID"], df["dayofweek"], df["hour_of_day"]):
            preds.append(self.profile_.get((int(loc), int(dow), int(hour)), self.global_mean_))
        return np.array(preds, dtype=float)


def evaluate_baselines_on_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_minutes: int,
) -> Dict[str, Dict[str, float]]:
    """Evaluate all baselines on held-out test split for a given horizon."""
    y_test = test_df["target"].to_numpy()
    results = {}

    # 1. Same hour last week (lag-168 relative to target)
    y_lag168 = BaselineForecaster.predict_lag_target_168(test_df)
    results["Baseline: Same Hour Last Week (target lag-168)"] = calculate_metrics(y_test, y_lag168)

    # 2. Same hour yesterday (lag-24 relative to target)
    y_lag24 = BaselineForecaster.predict_lag_target_24(test_df)
    results["Baseline: Same Hour Yesterday (target lag-24)"] = calculate_metrics(y_test, y_lag24)

    # 3. Persistence (origin demand)
    y_persist = BaselineForecaster.predict_persistence_origin(test_df)
    results["Baseline: Origin Persistence (origin lag-0)"] = calculate_metrics(y_test, y_persist)

    # 4. Rolling 24h average
    y_rolling = BaselineForecaster.predict_rolling_mean_24(test_df)
    results["Baseline: 24h Rolling Mean"] = calculate_metrics(y_test, y_rolling)

    # 5. Historical profile (zone x dow x hour)
    profile = HistoricalProfileBaseline().fit(train_df)
    y_profile = profile.predict(test_df)
    results["Baseline: Historical Zone-DoW-Hour Profile"] = calculate_metrics(y_test, y_profile)

    return results
