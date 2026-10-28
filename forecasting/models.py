"""Multi-horizon demand forecasting models and artifact management."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from forecasting.config import (
    ARTIFACTS_DIR,
    DEFAULT_HORIZONS_MINUTES,
    HORIZON_STEPS,
    MODEL_VERSION,
    MULTI_HORIZON_FEATURE_ORDER,
)

logger = logging.getLogger(__name__)


class MultiHorizonDemandModel:
    """Direct multi-horizon demand forecasting model.
    
    Fits separate, leak-free estimators for each prediction horizon (e.g. 60, 180, 360 minutes ahead).
    """

    def __init__(
        self,
        horizons_minutes: Optional[List[int]] = None,
        model_type: str = "lightgbm",
        model_params: Optional[Dict] = None,
    ):
        self.horizons_minutes = horizons_minutes or DEFAULT_HORIZONS_MINUTES
        self.model_type = model_type
        self.model_params = model_params or {}
        self.models: Dict[int, Union[lgb.LGBMRegressor, RandomForestRegressor]] = {}
        self.feature_names: List[str] = MULTI_HORIZON_FEATURE_ORDER
        self.feature_importances: Dict[int, Dict[str, float]] = {}
        self.training_metadata: Dict = {}

    def _create_estimator(self) -> Union[lgb.LGBMRegressor, RandomForestRegressor]:
        if self.model_type == "lightgbm":
            default_params = {
                "n_estimators": 100,
                "learning_rate": 0.08,
                "num_leaves": 31,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            default_params.update(self.model_params)
            return lgb.LGBMRegressor(**default_params)
        elif self.model_type == "random_forest":
            default_params = {
                "n_estimators": 100,
                "random_state": 42,
                "n_jobs": -1,
            }
            default_params.update(self.model_params)
            return RandomForestRegressor(**default_params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def fit_horizon(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        horizon_minutes: int,
    ) -> None:
        """Fit the direct estimator for a single forecast horizon."""
        logger.info(f"Training {self.model_type} estimator for horizon {horizon_minutes}m ({len(X_train)} samples)...")
        estimator = self._create_estimator()
        estimator.fit(X_train[self.feature_names], y_train)
        self.models[horizon_minutes] = estimator

        # Extract feature importances
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            total = np.sum(importances)
            normalized = (importances / total) if total > 0 else importances
            self.feature_importances[horizon_minutes] = {
                feat: round(float(imp), 5)
                for feat, imp in sorted(zip(self.feature_names, normalized), key=lambda x: x[1], reverse=True)
            }

    def predict(
        self,
        X: pd.DataFrame,
        horizon_minutes: int,
    ) -> np.ndarray:
        """Generate point forecasts for a specific horizon.
        
        Applies physical non-negativity constraint (demand >= 0.0).
        """
        if horizon_minutes not in self.models:
            raise KeyError(
                f"No estimator trained for horizon {horizon_minutes}m. "
                f"Available horizons: {list(self.models.keys())}"
            )
        estimator = self.models[horizon_minutes]
        preds = estimator.predict(X[self.feature_names])
        return np.maximum(preds, 0.0)

    def save(self, directory: Optional[Path] = None) -> Path:
        """Persist model estimators and metadata to disk."""
        save_dir = directory or ARTIFACTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        for horizon, estimator in self.models.items():
            model_path = save_dir / f"model_h{horizon}.joblib"
            joblib.dump(estimator, model_path)

        meta = {
            "model_version": MODEL_VERSION,
            "model_type": self.model_type,
            "horizons_minutes": self.horizons_minutes,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "training_metadata": self.training_metadata,
        }
        with open(save_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"MultiHorizonDemandModel saved to {save_dir}")
        return save_dir

    @classmethod
    def load(cls, directory: Optional[Path] = None) -> "MultiHorizonDemandModel":
        """Load persisted models and metadata from disk."""
        load_dir = directory or ARTIFACTS_DIR
        meta_path = load_dir / "model_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found at {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            horizons_minutes=meta["horizons_minutes"],
            model_type=meta["model_type"],
        )
        instance.feature_names = meta["feature_names"]
        instance.feature_importances = meta.get("feature_importances", {})
        instance.training_metadata = meta.get("training_metadata", {})

        for horizon in instance.horizons_minutes:
            model_file = load_dir / f"model_h{horizon}.joblib"
            if model_file.exists():
                instance.models[horizon] = joblib.load(model_file)
            else:
                logger.warning(f"Model file for horizon {horizon}m not found at {model_file}")

        return instance
