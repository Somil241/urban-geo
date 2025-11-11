"""
Feature engineering for traffic prediction models.
Creates spatial, temporal, historical, and lagged features.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional
import logging
from config import DB_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for traffic prediction."""
    
    def __init__(self):
        self.db_url = self._get_db_url()
        self.engine = create_engine(self.db_url)
    
    def _get_db_url(self) -> str:
        """Construct database URL from config."""
        return (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
    
    def load_traffic_data(self, days_back: int = 30, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load traffic readings from CSV file or database.

        Args:
            days_back: Number of days of historical data to load
            csv_path: Path to CSV file (if None, loads from database)

        Returns:
            DataFrame with traffic data
        """
        if csv_path:
            # Load from CSV
            df = pd.read_csv(csv_path)

            # Convert column names to match expected format
            df = df.rename(columns={
                'LocalDateTime': 'timestamp',
                'TrafficIndexLive': 'traffic_index',
                'TravelTimeLivePer10KmsMins': 'travel_time_live',
                'TravelTimeHistoricPer10KmsMins': 'travel_time_historic',
                'JamsDelay': 'jams_delay',
                'JamsCount': 'jams_count',
                'JamsLengthInKms': 'jams_length',
                'MinsDelay': 'mins_delay',
                'TrafficIndexWeekAgo': 'traffic_index_week_ago',
                'City': 'city',
                'Country': 'country'
            })

            # Parse datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Calculate speed from travel time (inverse relationship)
            # Assuming average speed = (10 km / travel_time_per_10km) * 60 mins/hour
            df['speed'] = 600.0 / (df['travel_time_live'] + 1e-6)  # km/h
            df['free_flow_speed'] = 600.0 / (df['travel_time_historic'] + 1e-6)  # km/h

            # Extract temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek

            # Add placeholder lat/lon (you'll need actual city coordinates)
            # Using city-based grouping for now
            city_coords = {
                'abu-dhabi': (24.4539, 54.3773),
                # Add more cities as needed
            }

            df['lat'] = df['city'].map(lambda x: city_coords.get(x, (0, 0))[0])
            df['lon'] = df['city'].map(lambda x: city_coords.get(x, (0, 0))[1])

            # Add confidence and road_class placeholders
            df['confidence'] = 0.8
            df['road_class'] = 'urban'

            # Add unique ID
            df['id'] = range(len(df))

            # Filter by date
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df['timestamp'] >= cutoff_date]

            # Drop columns not needed for modeling
            columns_to_drop = ['country', 'city', 'UpdateTimeUTC', 'LocalDateTimeWeekAgo',
                             'traffic_index', 'travel_time_live', 'travel_time_historic',
                             'jams_delay', 'jams_count', 'jams_length', 'mins_delay',
                             'traffic_index_week_ago']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            logger.info(f"Loaded {len(df)} traffic readings from CSV")
            return df

        else:
            # Load from database (original logic)
            cutoff_date = datetime.now() - timedelta(days=days_back)

            query = text("""
                SELECT
                    id, lat, lon, speed, confidence, timestamp,
                    day_of_week, hour, free_flow_speed, road_class
                FROM traffic_readings
                WHERE timestamp >= :cutoff_date
                ORDER BY timestamp
            """)

            df = pd.read_sql(query, self.engine, params={"cutoff_date": cutoff_date})
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            logger.info(f"Loaded {len(df)} traffic readings from database")
            return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with added temporal features
        """
        df = df.copy()
        
        # Hour features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        
        # Day of week features
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Weekend indicator
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        
        # Rush hour indicators (morning: 7-9, evening: 17-19)
        df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] < 9)).astype(int)
        df["is_evening_rush"] = ((df["hour"] >= 17) & (df["hour"] < 19)).astype(int)
        df["is_rush_hour"] = (df["is_morning_rush"] | df["is_evening_rush"]).astype(int)
        
        # Time of day category
        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"]
        )
        df["time_of_day"] = df["time_of_day"].astype(str)
        
        return df
    
    def create_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spatial features.
        
        Args:
            df: DataFrame with lat/lon columns
            
        Returns:
            DataFrame with added spatial features
        """
        df = df.copy()
        
        # Round coordinates to create spatial bins
        df["lat_bin"] = np.round(df["lat"], 2)
        df["lon_bin"] = np.round(df["lon"], 2)
        
        # Calculate nearby road density (simplified - count nearby points)
        df["nearby_points"] = 0  # Placeholder - would use PostGIS for real implementation
        
        return df
    
    def create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create historical features.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with added historical features
        """
        df = df.copy()
        
        # Group by location and calculate historical averages
        location_cols = ["lat_bin", "lon_bin"]
        
        # Average speed at same hour
        df["avg_speed_same_hour"] = df.groupby(location_cols + ["hour"])["speed"].transform("mean")
        
        # Average speed at same day of week and hour
        df["avg_speed_same_dow_hour"] = df.groupby(location_cols + ["day_of_week", "hour"])["speed"].transform("mean")
        
        # Rolling averages (last 7 days)
        df = df.sort_values(["lat_bin", "lon_bin", "timestamp"])
        
        # 7-day rolling average
        df["avg_speed_7d"] = df.groupby(location_cols)["speed"].transform(
            lambda x: x.rolling(window=7*24, min_periods=1).mean()
        )
        
        # Free flow speed ratio
        df["speed_ratio"] = df["speed"] / (df["free_flow_speed"] + 1e-6)
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for time-series prediction.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            DataFrame with added lagged features
        """
        df = df.copy()
        df = df.sort_values(["lat_bin", "lon_bin", "timestamp"])
        
        # Group by location
        location_cols = ["lat_bin", "lon_bin"]
        
        # Lagged speeds (15 min, 30 min, 1 hour ago)
        # Assuming ~15 min intervals, approximate with shifts
        df["speed_15min_ago"] = df.groupby(location_cols)["speed"].shift(1)
        df["speed_30min_ago"] = df.groupby(location_cols)["speed"].shift(2)
        df["speed_1h_ago"] = df.groupby(location_cols)["speed"].shift(4)
        
        # Fill missing values with forward fill
        df["speed_15min_ago"] = df.groupby(location_cols)["speed_15min_ago"].ffill()
        df["speed_30min_ago"] = df.groupby(location_cols)["speed_30min_ago"].ffill()
        df["speed_1h_ago"] = df.groupby(location_cols)["speed_1h_ago"].ffill()
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, minutes_ahead: int = 0) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: DataFrame with traffic data
            minutes_ahead: Minutes ahead to predict (0 for current, 30 for future)
            
        Returns:
            DataFrame with target variable
        """
        df = df.copy()
        df = df.sort_values(["lat_bin", "lon_bin", "timestamp"])
        
        location_cols = ["lat_bin", "lon_bin"]
        
        if minutes_ahead == 0:
            # Current prediction - use current speed
            df["target_speed"] = df["speed"]
        else:
            # Future prediction - shift forward by number of periods
            # Assuming ~15 min intervals
            periods = max(1, minutes_ahead // 15)
            df["target_speed"] = df.groupby(location_cols)["speed"].shift(-periods)
        
        # Remove rows with missing target
        df = df.dropna(subset=["target_speed"])
        
        return df
    
    def engineer_features(self, days_back: int = 30, minutes_ahead: int = 0, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            days_back: Number of days of historical data
            minutes_ahead: Minutes ahead for prediction (0 or 30)
            csv_path: Path to CSV file (if None, loads from database)

        Returns:
            DataFrame with all features
        """
        logger.info("Loading traffic data...")
        df = self.load_traffic_data(days_back, csv_path=csv_path)

        if df.empty:
            logger.warning("No data available")
            return df

        logger.info("Creating temporal features...")
        df = self.create_temporal_features(df)

        logger.info("Creating spatial features...")
        df = self.create_spatial_features(df)

        logger.info("Creating historical features...")
        df = self.create_historical_features(df)

        logger.info("Creating lagged features...")
        df = self.create_lagged_features(df)

        logger.info("Creating target variable...")
        df = self.create_target_variable(df, minutes_ahead)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        return [
            # Spatial
            "lat", "lon", "lat_bin", "lon_bin",
            # Temporal
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "is_weekend", "is_morning_rush", "is_evening_rush", "is_rush_hour",
            # Historical
            "avg_speed_same_hour", "avg_speed_same_dow_hour", "avg_speed_7d",
            "speed_ratio", "free_flow_speed",
            # Lagged
            "speed_15min_ago", "speed_30min_ago", "speed_1h_ago",
            # Other
            "confidence", "road_class"
        ]


if __name__ == "__main__":
    # Test feature engineering
    engineer = FeatureEngineer()
    df = engineer.engineer_features(days_back=7, minutes_ahead=0)
    print(f"\nFeature engineering test complete. Shape: {df.shape}")
    print(f"\nFeatures: {engineer.get_feature_columns()}")

