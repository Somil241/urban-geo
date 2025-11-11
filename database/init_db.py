"""
Database initialization script for Urban-Geo.
Creates tables and sets up PostGIS extensions.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from config import DB_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_url():
    """Construct database URL from config."""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def init_database():
    """Initialize database schema."""
    engine = create_engine(get_db_url())
    
    # Read and execute schema SQL
    schema_file = Path(__file__).parent / "schema.sql"
    
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        return False
    
    with open(schema_file, 'r') as f:
        schema_sql = f.read()
    
    try:
        with engine.begin() as conn:
            # Execute schema SQL
            conn.execute(text(schema_sql))
            logger.info("✅ Database schema initialized successfully")
        
        # Verify PostGIS extension
        with engine.connect() as conn:
            result = conn.execute(text("SELECT PostGIS_version();"))
            version = result.scalar()
            logger.info(f"✅ PostGIS version: {version}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        return False


if __name__ == "__main__":
    init_database()

