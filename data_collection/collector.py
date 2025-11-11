"""
Enhanced data collection layer for Urban-Geo.
Implements TomTom API integration with rate limiting and scheduled collection.
"""
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from config import DB_CONFIG, TOMTOM_API_KEY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Rate limiter for TomTom API (2,500 requests/day)."""
    daily_limit: int = 2500
    requests_today: int = 0
    last_reset: datetime = None
    
    def __post_init__(self):
        if self.last_reset is None:
            self.last_reset = datetime.now(timezone.utc)
    
    def can_make_request(self) -> bool:
        """Check if we can make a request."""
        now = datetime.now(timezone.utc)
        # Reset daily counter if new day
        if (now - self.last_reset).days >= 1:
            self.requests_today = 0
            self.last_reset = now
        
        return self.requests_today < self.daily_limit
    
    def record_request(self):
        """Record that a request was made."""
        self.requests_today += 1
    
    def get_remaining_requests(self) -> int:
        """Get remaining requests for today."""
        return max(0, self.daily_limit - self.requests_today)


class TrafficDataCollector:
    """Collects traffic data from TomTom API."""
    
    def __init__(self):
        self.api_key = TOMTOM_API_KEY
        self.rate_limiter = RateLimiter()
        self.base_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
        self.db_url = self._get_db_url()
        self.engine = create_engine(self.db_url)
    
    def _get_db_url(self) -> str:
        """Construct database URL from config."""
        return (
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
    
    def fetch_tomtom_data(self, lat: float, lon: float, retries: int = 5) -> Optional[Dict]:
        """
        Fetch traffic data from TomTom API with retry logic.
        
        Args:
            lat: Latitude
            lon: Longitude
            retries: Number of retry attempts
            
        Returns:
            API response JSON or None if failed
        """
        if not self.rate_limiter.can_make_request():
            logger.warning(f"Rate limit reached. Remaining: {self.rate_limiter.get_remaining_requests()}")
            return None
        
        params = {
            "key": self.api_key,
            "point": f"{lat},{lon}"
        }
        
        backoff = 1
        for attempt in range(retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    self.rate_limiter.record_request()
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    logger.warning(f"Rate limit exceeded. Waiting {backoff * 60}s...")
                    time.sleep(backoff * 60)
                    backoff *= 2
                else:
                    logger.warning(f"API error {response.status_code}, retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    
            except Exception as e:
                logger.error(f"Error fetching data: {e}, retrying in {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
        
        logger.error(f"Failed to fetch data after {retries} retries for ({lat}, {lon})")
        return None
    
    def process_api_data(self, api_data: Dict, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """
        Process API response into DataFrame.
        
        Args:
            api_data: API response JSON
            lat: Latitude
            lon: Longitude
            
        Returns:
            DataFrame with processed data
        """
        if not api_data or "flowSegmentData" not in api_data:
            return None
        
        segment = api_data["flowSegmentData"]
        now = datetime.now(timezone.utc)
        
        current_speed = segment.get("currentSpeed", 0)
        free_flow_speed = segment.get("freeFlowSpeed", 0)
        confidence = segment.get("confidence", 0)
        road_class = segment.get("frc", "UNKNOWN")
        
        df = pd.DataFrame([{
            "lat": lat,
            "lon": lon,
            "speed": current_speed,
            "confidence": confidence,
            "timestamp": now,
            "day_of_week": now.weekday(),  # 0=Monday, 6=Sunday
            "hour": now.hour,
            "free_flow_speed": free_flow_speed,
            "road_class": road_class,
            "source": "TomTom"
        }])
        
        return df
    
    def save_to_database(self, df: pd.DataFrame) -> bool:
        """
        Save traffic readings to database.
        
        Args:
            df: DataFrame with traffic data
            
        Returns:
            True if successful, False otherwise
        """
        if df is None or df.empty:
            logger.warning("No data to save")
            return False
        
        try:
            # Use text() for raw SQL to ensure geometry is set correctly
            with self.engine.begin() as conn:
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO traffic_readings 
                        (lat, lon, speed, confidence, timestamp, day_of_week, hour, 
                         free_flow_speed, road_class, source)
                        VALUES 
                        (:lat, :lon, :speed, :confidence, :timestamp, :day_of_week, :hour,
                         :free_flow_speed, :road_class, :source)
                    """), {
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "speed": row["speed"],
                        "confidence": row["confidence"],
                        "timestamp": row["timestamp"],
                        "day_of_week": row["day_of_week"],
                        "hour": row["hour"],
                        "free_flow_speed": row.get("free_flow_speed"),
                        "road_class": row.get("road_class"),
                        "source": row.get("source", "TomTom")
                    })
            
            logger.info(f"✅ Saved {len(df)} record(s) to database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving to database: {e}")
            return False
    
    def collect_for_location(self, lat: float, lon: float) -> bool:
        """
        Collect traffic data for a single location.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Collecting data for ({lat}, {lon})")
        
        api_data = self.fetch_tomtom_data(lat, lon)
        if api_data is None:
            return False
        
        df = self.process_api_data(api_data, lat, lon)
        if df is None:
            logger.warning("No valid data processed")
            return False
        
        return self.save_to_database(df)
    
    def collect_for_locations(self, locations: List[Tuple[float, float]], delay: float = 1.0) -> Dict[str, int]:
        """
        Collect traffic data for multiple locations.
        
        Args:
            locations: List of (lat, lon) tuples
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary with success/failure counts
        """
        results = {"success": 0, "failed": 0, "rate_limited": 0}
        
        for lat, lon in locations:
            if not self.rate_limiter.can_make_request():
                logger.warning("Daily rate limit reached. Stopping collection.")
                results["rate_limited"] = len(locations) - results["success"] - results["failed"]
                break
            
            if self.collect_for_location(lat, lon):
                results["success"] += 1
            else:
                results["failed"] += 1
            
            # Rate limiting delay
            time.sleep(delay)
        
        logger.info(f"Collection complete: {results}")
        return results
    
    def get_active_locations(self) -> List[Tuple[float, float, str]]:
        """
        Get active monitoring locations from database.
        
        Returns:
            List of (lat, lon, name) tuples
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT lat, lon, location_name 
                    FROM prediction_locations 
                    WHERE is_active = TRUE
                    ORDER BY priority DESC
                """))
                return [(row[0], row[1], row[2] or "") for row in result]
        except Exception as e:
            logger.error(f"Error fetching locations: {e}")
            return []


def main():
    """Main function for manual data collection."""
    collector = TrafficDataCollector()
    
    # Example: Collect for a single location
    # collector.collect_for_location(28.6139, 77.2090)  # Delhi
    
    # Or collect for all active locations
    locations = collector.get_active_locations()
    if locations:
        logger.info(f"Found {len(locations)} active locations")
        collector.collect_for_locations([(lat, lon) for lat, lon, _ in locations])
    else:
        logger.warning("No active locations found. Please add locations to prediction_locations table.")


if __name__ == "__main__":
    main()

