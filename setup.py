#!/usr/bin/env python3
"""
Quick start script for Urban-Geo.
Sets up database, adds default locations, and starts data collection.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from database.init_db import init_database
from utils.add_locations import add_default_locations
from data_collection.collector import TrafficDataCollector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Quick start setup."""
    logger.info("🚀 Starting Urban-Geo setup...")
    
    # Step 1: Initialize database
    logger.info("Step 1: Initializing database...")
    if not init_database():
        logger.error("Failed to initialize database. Check your configuration.")
        return False
    
    # Step 2: Add default locations
    logger.info("Step 2: Adding default monitoring locations...")
    try:
        add_default_locations()
    except Exception as e:
        logger.warning(f"Warning adding locations: {e}")
    
    # Step 3: Test data collection
    logger.info("Step 3: Testing data collection...")
    try:
        collector = TrafficDataCollector()
        locations = collector.get_active_locations()
        if locations:
            logger.info(f"Found {len(locations)} active locations")
            logger.info("Running test collection (first location only)...")
            if locations:
                lat, lon, name = locations[0]
                collector.collect_for_location(lat, lon)
                logger.info("✅ Test collection successful!")
        else:
            logger.warning("No active locations found. Add locations first.")
    except Exception as e:
        logger.error(f"Error testing collection: {e}")
    
    logger.info("\n✅ Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Train model: python ml/model_training.py")
    logger.info("2. Start API server: python run_server.py")
    logger.info("3. Set up scheduled collection (cron or Celery)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

