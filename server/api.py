"""
FastAPI service for Urban-Geo traffic prediction API.
Provides endpoints for current and future traffic predictions.
"""
import sys
from pathlib import Path

# Add project root to Python path to ensure imports work
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import logging
import time
from contextlib import asynccontextmanager

from ml.model_training import TrafficPredictor
from config import DB_CONFIG
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
current_predictor: Optional[TrafficPredictor] = None
future_predictor: Optional[TrafficPredictor] = None
MODEL_DIR = Path(__file__).parent.parent / "models"


def get_latest_model_version(model_type: str = "current") -> Optional[str]:
    """Get the latest model version from the models directory."""
    try:
        model_files = list(MODEL_DIR.glob(f"{model_type}_*.pkl"))
        if not model_files:
            return None
        # Sort by modification time and get the latest
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        # Extract version from filename (remove .pkl extension)
        return latest_model.stem
    except Exception as e:
        logger.error(f"Error finding latest model: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global current_predictor, future_predictor

    logger.info("Loading models...")
    try:
        # Load current traffic model
        current_predictor = TrafficPredictor(model_type="current")
        latest_version = get_latest_model_version("current")

        if latest_version:
            current_predictor.load_model(latest_version)
            logger.info(f"Loaded current model: {latest_version}")
        else:
            logger.warning("No trained current model found")

        # Load future prediction model (if exists)
        future_predictor = TrafficPredictor(model_type="future_30min")
        future_version = get_latest_model_version("future_30min")

        if future_version:
            future_predictor.load_model(future_version)
            logger.info(f"Loaded future model: {future_version}")
        else:
            logger.warning("No trained future model found, will use current model")

        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Urban-Geo Traffic Prediction API",
    description="Real-time traffic prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class FuturePredictionRequest(PredictionRequest):
    minutes_ahead: int = Field(default=30, ge=0, le=60, description="Minutes ahead for prediction")


class PredictionResponse(BaseModel):
    traffic_level: str = Field(..., description="Traffic level: low, medium, high")
    speed_kmh: float = Field(..., description="Predicted speed in km/h")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class FuturePredictionResponse(BaseModel):
    predicted_traffic_level: str = Field(..., description="Predicted traffic level")
    predicted_speed_kmh: float = Field(..., description="Predicted speed in km/h")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")


class StatsResponse(BaseModel):
    total_readings: int
    locations_monitored: int
    oldest_data: Optional[str]
    newest_data: Optional[str]
    avg_speed_kmh: Optional[float]


# Database connection
def get_db_engine():
    db_url = (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    return create_engine(db_url)


def get_predictor(model_type: str = "current") -> TrafficPredictor:
    """Get or create predictor instance."""
    global current_predictor, future_predictor

    if model_type == "current":
        if current_predictor is None or current_predictor.model is None:
            current_predictor = TrafficPredictor(model_type="current")
            # Try to load latest model
            latest_version = get_latest_model_version("current")
            if latest_version:
                current_predictor.load_model(latest_version)
                logger.info(f"Loaded current model on demand: {latest_version}")
            else:
                raise HTTPException(status_code=503, detail="No trained model available")
        return current_predictor
    else:
        if future_predictor is None or future_predictor.model is None:
            future_predictor = TrafficPredictor(model_type="future_30min")
            # Try to load latest model, fall back to current model
            latest_version = get_latest_model_version("future_30min")
            if latest_version:
                future_predictor.load_model(latest_version)
                logger.info(f"Loaded future model on demand: {latest_version}")
            else:
                # Fall back to current model
                latest_version = get_latest_model_version("current")
                if latest_version:
                    future_predictor = current_predictor if current_predictor else TrafficPredictor(model_type="current")
                    if future_predictor.model is None:
                        future_predictor.load_model(latest_version)
                    logger.info("Using current model for future predictions")
                else:
                    raise HTTPException(status_code=503, detail="No trained model available")
        return future_predictor


def log_api_request(endpoint: str, lat: float, lon: float, status_code: int, response_time_ms: int, error: Optional[str] = None):
    """Log API request to database."""
    try:
        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO api_logs (endpoint, lat, lon, response_time_ms, status_code, error_message)
                VALUES (:endpoint, :lat, :lon, :response_time_ms, :status_code, :error)
            """), {
                "endpoint": endpoint,
                "lat": lat,
                "lon": lon,
                "response_time_ms": response_time_ms,
                "status_code": status_code,
                "error": error
            })
    except Exception as e:
        logger.error(f"Error logging API request: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": current_predictor is not None
    }


@app.post("/predict/current", response_model=PredictionResponse)
async def predict_current(request: PredictionRequest):
    """
    Predict current traffic conditions for a location.
    
    Args:
        request: Prediction request with lat/lon
        
    Returns:
        Prediction response with traffic level, speed, and confidence
    """
    start_time = time.time()
    
    try:
        predictor = get_predictor("current")
        result = predictor.predict(request.lat, request.lon)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        log_api_request("/predict/current", request.lat, request.lon, 200, response_time_ms)
        
        return PredictionResponse(
            traffic_level=result["traffic_level"],
            speed_kmh=result["predicted_speed_kmh"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error in predict_current: {e}")
        log_api_request("/predict/current", request.lat, request.lon, 500, response_time_ms, str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/future", response_model=FuturePredictionResponse)
async def predict_future(request: FuturePredictionRequest):
    """
    Predict traffic conditions for a location N minutes ahead.
    
    Args:
        request: Prediction request with lat/lon and minutes_ahead
        
    Returns:
        Prediction response with predicted traffic level, speed, and confidence
    """
    start_time = time.time()
    
    try:
        # For now, use same model (in production, use specialized future model)
        predictor = get_predictor("future_30min" if request.minutes_ahead == 30 else "current")
        result = predictor.predict(request.lat, request.lon)
        
        response_time_ms = int((time.time() - start_time) * 1000)
        
        log_api_request("/predict/future", request.lat, request.lon, 200, response_time_ms)
        
        return FuturePredictionResponse(
            predicted_traffic_level=result["traffic_level"],
            predicted_speed_kmh=result["predicted_speed_kmh"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        response_time_ms = int((time.time() - start_time) * 1000)
        logger.error(f"Error in predict_future: {e}")
        log_api_request("/predict/future", request.lat, request.lon, 500, response_time_ms, str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get historical data statistics.
    
    Returns:
        Statistics about collected data
    """
    try:
        engine = get_db_engine()
        
        with engine.connect() as conn:
            # Total readings
            total_readings = conn.execute(text("SELECT COUNT(*) FROM traffic_readings")).scalar()
            
            # Locations monitored
            locations = conn.execute(text("SELECT COUNT(*) FROM prediction_locations WHERE is_active = TRUE")).scalar()
            
            # Oldest and newest data
            oldest = conn.execute(text("SELECT MIN(timestamp) FROM traffic_readings")).scalar()
            newest = conn.execute(text("SELECT MAX(timestamp) FROM traffic_readings")).scalar()
            
            # Average speed
            avg_speed = conn.execute(text("SELECT AVG(speed) FROM traffic_readings")).scalar()
        
        return StatsResponse(
            total_readings=int(total_readings) if total_readings else 0,
            locations_monitored=int(locations) if locations else 0,
            oldest_data=oldest.isoformat() if oldest else None,
            newest_data=newest.isoformat() if newest else None,
            avg_speed_kmh=float(avg_speed) if avg_speed else None
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

