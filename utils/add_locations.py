"""
Script to add prediction locations to the database.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from config import DB_CONFIG
from utils.helpers import get_db_url, validate_coordinates
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_location(lat: float, lon: float, name: str, priority: int = 1):
    """
    Add a prediction location to the database.
    
    Args:
        lat: Latitude
        lon: Longitude
        name: Location name
        priority: Priority level (higher = more frequent polling)
    """
    if not validate_coordinates(lat, lon):
        raise ValueError(f"Invalid coordinates: ({lat}, {lon})")
    
    db_url = get_db_url(DB_CONFIG)
    engine = create_engine(db_url)
    
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO prediction_locations (lat, lon, location_name, priority, is_active)
                VALUES (:lat, :lon, :name, :priority, TRUE)
                ON CONFLICT (lat, lon) 
                DO UPDATE SET 
                    location_name = :name,
                    priority = :priority,
                    is_active = TRUE,
                    updated_at = NOW()
            """), {
                "lat": lat,
                "lon": lon,
                "name": name,
                "priority": priority
            })
        logger.info(f"✅ Added location: {name} ({lat}, {lon})")
    except Exception as e:
        logger.error(f"❌ Error adding location: {e}")
        raise


def add_default_locations():
    """Add some default locations for testing."""
    locations = [
        # Delhi, India locations
        (28.6139, 77.2090, "Connaught Place", 2),
        (28.5562, 77.1000, "IGI Airport", 2),
        (28.7041, 77.1025, "Delhi University", 1),
        (28.5355, 77.3910, "Gurgaon Sector 29", 1),
        (28.6139, 77.2290, "Rajiv Chowk", 2),
    ]
    
    for lat, lon, name, priority in locations:
        try:
            add_location(lat, lon, name, priority)
        except Exception as e:
            logger.warning(f"Failed to add {name}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add prediction location")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--name", type=str, help="Location name")
    parser.add_argument("--priority", type=int, default=1, help="Priority (1-5)")
    parser.add_argument("--default", action="store_true", help="Add default locations")
    
    args = parser.parse_args()
    
    if args.default:
        add_default_locations()
    elif args.lat and args.lon and args.name:
        add_location(args.lat, args.lon, args.name, args.priority)
    else:
        parser.print_help()

