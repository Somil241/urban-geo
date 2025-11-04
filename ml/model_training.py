"""
Machine learning model training for traffic prediction.
Uses LightGBM for regression and classification tasks.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import lightgbm as lgb
import joblib
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
import json

from ml.feature_engineering import FeatureEngineer
from config import DB_CONFIG
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficPredictor:
    """Traffic prediction model trainer and predictor."""
    
    def __init__(self, model_type: str = "current", csv_path: Optional[str] = None):
        """
        Initialize predictor.

        Args:
            model_type: 'current' or 'future_30min'
            csv_path: Path to CSV file (if None, loads from database)
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.feature_engineer = FeatureEngineer()
        self.csv_path = csv_path
        self.model_dir = Path(__file__).parent.parent / "models"
        self.model_dir.mkdir(exist_ok=True)
    
    def prepare_data(self, days_back: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data.

        Args:
            days_back: Number of days of historical data

        Returns:
            X (features), y (target)
        """
        minutes_ahead = 0 if self.model_type == "current" else 30

        df = self.feature_engineer.engineer_features(
            days_back=days_back,
            minutes_ahead=minutes_ahead,
            csv_path=self.csv_path
        )
        
        if df.empty:
            raise ValueError("No data available for training")
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_columns()
        
        # Select only available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Handle categorical columns
        categorical_cols = ["road_class", "time_of_day"]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # One-hot encode categorical features
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
            # Update available cols
            available_cols = [col for col in df.columns if col not in ["id", "target_speed", "timestamp", "speed"]]
        
        self.feature_columns = available_cols
        
        X = df[available_cols]
        y = df["target_speed"]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Prepared data: {len(X)} samples, {len(available_cols)} features")
        return X, y
    
    def train(
        self,
        days_back: int = 30,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            days_back: Number of days of historical data
            test_size: Test set size ratio
            random_state: Random seed
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Prepare data
        X, y = self.prepare_data(days_back)
        
        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples. Need at least 100.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train LightGBM model
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": random_state
        }
        
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)]
        )
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        logger.info(f"Training complete. Test MAE: {metrics['test_mae']:.2f} km/h")
        logger.info(f"Test RMSE: {metrics['test_rmse']:.2f} km/h")
        logger.info(f"Test R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def save_model(self, version: Optional[str] = None) -> str:
        """
        Save trained model.
        
        Args:
            version: Model version identifier (auto-generated if None)
            
        Returns:
            Model version string
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if version is None:
            version = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_path = self.model_dir / f"{version}.pkl"
        metadata_path = self.model_dir / f"{version}_metadata.json"
        
        # Save model
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "model_type": self.model_type
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            "version": version,
            "model_type": self.model_type,
            "feature_columns": self.feature_columns,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return version
    
    def load_model(self, version: str):
        """
        Load trained model.
        
        Args:
            version: Model version identifier
        """
        model_path = self.model_dir / f"{version}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.model_type = model_data["model_type"]
        
        logger.info(f"Model loaded: {version}")
    
    def predict(self, lat: float, lon: float, timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Make prediction for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Prediction timestamp (defaults to now)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train model first.")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create feature vector
        # This is simplified - in production, you'd fetch historical data and compute features
        features = self._create_feature_vector(lat, lon, timestamp)
        
        # Predict
        predicted_speed = self.model.predict([features])[0]
        
        # Classify traffic level
        traffic_level = self._classify_traffic_level(predicted_speed)
        
        # Calculate confidence (simplified - use model's prediction variance in production)
        confidence = 0.8  # Placeholder
        
        return {
            "predicted_speed_kmh": float(predicted_speed),
            "traffic_level": traffic_level,
            "confidence": confidence
        }
    
    def _create_feature_vector(self, lat: float, lon: float, timestamp: datetime) -> np.ndarray:
        """
        Create feature vector for prediction.
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Timestamp
            
        Returns:
            Feature vector
        """
        # Simplified feature creation - in production, fetch historical data
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        features_dict = {
            "lat": lat,
            "lon": lon,
            "lat_bin": np.round(lat, 2),
            "lon_bin": np.round(lon, 2),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day_of_week / 7),
            "day_cos": np.cos(2 * np.pi * day_of_week / 7),
            "is_weekend": 1 if day_of_week >= 5 else 0,
            "is_morning_rush": 1 if 7 <= hour < 9 else 0,
            "is_evening_rush": 1 if 17 <= hour < 19 else 0,
            "is_rush_hour": 1 if (7 <= hour < 9) or (17 <= hour < 19) else 0,
            "avg_speed_same_hour": 50.0,  # Placeholder - fetch from DB
            "avg_speed_same_dow_hour": 50.0,  # Placeholder
            "avg_speed_7d": 50.0,  # Placeholder
            "speed_ratio": 1.0,  # Placeholder
            "free_flow_speed": 60.0,  # Placeholder
            "speed_15min_ago": 50.0,  # Placeholder
            "speed_30min_ago": 50.0,  # Placeholder
            "speed_1h_ago": 50.0,  # Placeholder
            "confidence": 0.8  # Placeholder
        }
        
        # Create array matching feature columns
        feature_vector = []
        for col in self.feature_columns:
            if col in features_dict:
                feature_vector.append(features_dict[col])
            else:
                # Handle one-hot encoded columns
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def _classify_traffic_level(self, speed: float) -> str:
        """
        Classify traffic level based on speed.
        
        Args:
            speed: Predicted speed in km/h
            
        Returns:
            Traffic level: 'low', 'medium', 'high'
        """
        if speed >= 50:
            return "low"
        elif speed >= 30:
            return "medium"
        else:
            return "high"


if __name__ == "__main__":
    # Train current traffic model using CSV data
    csv_path = Path(__file__).parent.parent / "database" / "ForExportNewApi.csv"

    predictor = TrafficPredictor(model_type="current", csv_path=str(csv_path))
    metrics = predictor.train(days_back=30)
    version = predictor.save_model()
    print(f"\nModel trained and saved: {version}")
    print(f"Metrics: {metrics}")

