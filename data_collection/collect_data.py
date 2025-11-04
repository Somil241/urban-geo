"""
Scheduled data collection using Celery Beat (optional) or cron.
For cron-based collection, use: */15 * * * * python collect_data.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_collection.collector import TrafficDataCollector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scheduled_collection():
    """Run scheduled data collection."""
    collector = TrafficDataCollector()
    
    # Get active locations
    locations = collector.get_active_locations()
    
    if not locations:
        logger.warning("No active locations configured. Skipping collection.")
        return
    
    logger.info(f"Starting scheduled collection for {len(locations)} locations")
    
    # Collect data with 1 second delay between requests
    results = collector.collect_for_locations(
        [(lat, lon) for lat, lon, _ in locations],
        delay=1.0
    )
    
    logger.info(f"Collection completed: {results['success']} succeeded, {results['failed']} failed")


if __name__ == "__main__":
    scheduled_collection()

