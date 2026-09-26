"""
Tabular ML Layer — Robust classification and regression for directional market probabilities.

Supports:
- Logistic Regression
- Random Forest
- Gradient Boosting / LightGBM / XGBoost
- Outputs: probability_up, probability_down, expected_return, expected_volatility, confidence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TabularMLModel:
    """Trains and predicts market probabilities without future leakage."""

    def __init__(self, model_type: str = "random_forest", target_horizon: int = 5):
        self.model_type = model_type.lower()
        self.target_horizon = target_horizon
        self.model = None
        self.feature_columns: List[str] = []
        self.is_trained = False

    def _init_model(self):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        if self.model_type == "logistic":
            return LogisticRegression(max_iter=500, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        elif self.model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(n_estimators=60, max_depth=3, random_state=42, verbose=-1)
            except Exception:
                return GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(n_estimators=60, max_depth=3, random_state=42, eval_metric="logloss")
            except Exception:
                return GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=80, max_depth=4, random_state=42)

    def prepare_data(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Creates time-safe forward label:
        target = 1 if close(t + target_horizon) > close(t) else 0.
        Drops last target_horizon rows to prevent look-ahead bias in evaluation.
        """
        df = df.copy()
        forward_ret = df["close"].shift(-self.target_horizon) / df["close"] - 1.0
        target = (forward_ret > 0.002).astype(int)  # 20 bps hurdle rate to cover costs

        # Exclude forward label and non-numeric columns
        non_feat = {"date", "symbol", "timestamp", "quality_status"}
        if feature_cols:
            cols = [c for c in feature_cols if c in df.columns and c not in non_feat]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cols = [c for c in numeric_cols if c not in non_feat and c != "target"]

        X = df[cols].iloc[:-self.target_horizon].fillna(0)
        y = target.iloc[:-self.target_horizon]
        return X, y

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train model on chronological dataset."""
        if df is None or len(df) <= self.target_horizon + 30:
            return {"status": "error", "message": "Insufficient bars for training"}

        X, y = self.prepare_data(df, feature_cols)
        if len(X) < 20 or y.nunique() < 2:
            return {"status": "error", "message": "Insufficient sample variance"}

        self.feature_columns = list(X.columns)
        self.model = self._init_model()
        self.model.fit(X, y)
        self.is_trained = True

        acc = float(self.model.score(X, y))
        return {
            "status": "trained",
            "model_type": self.model_type,
            "train_samples": len(X),
            "feature_count": len(self.feature_columns),
            "train_accuracy": round(acc, 3),
        }

    def predict_latest(self, features_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for the latest decision bar.
        Returns: probability_up, probability_down, expected_return, expected_volatility, confidence.
        """
        if not self.is_trained or self.model is None:
            return {
                "probability_up": 0.50,
                "probability_down": 0.50,
                "expected_return": 0.0,
                "expected_volatility": 0.02,
                "confidence": 0.40,
                "is_trained": False,
            }

        x_row = [float(features_dict.get(col, 0.0)) for col in self.feature_columns]
        X = pd.DataFrame([x_row], columns=self.feature_columns).fillna(0)

        probs = self.model.predict_proba(X)[0]
        prob_down = float(probs[0])
        prob_up = float(probs[1]) if len(probs) > 1 else (1.0 - prob_down)

        # Expected return estimation (heuristic mapping from probability differential)
        expected_ret = round((prob_up - 0.50) * 0.06, 4)  # ~3% max expected 5d return
        confidence = round(max(prob_up, prob_down), 3)
        exp_vol = float(features_dict.get("atr_pct", 2.0)) / 100.0

        return {
            "probability_up": round(prob_up, 3),
            "probability_down": round(prob_down, 3),
            "expected_return": expected_ret,
            "expected_volatility": round(exp_vol, 4),
            "confidence": confidence,
            "is_trained": True,
        }
