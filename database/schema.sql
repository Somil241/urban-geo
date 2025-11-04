-- Urban-Geo Database Schema with PostGIS Support
-- PostgreSQL 14+ with PostGIS extension required

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Traffic readings table (stores historical traffic data)
CREATE TABLE IF NOT EXISTS traffic_readings (
    id SERIAL PRIMARY KEY,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    location GEOMETRY(POINT, 4326), -- PostGIS geometry for spatial queries
    speed DOUBLE PRECISION NOT NULL, -- Current speed in km/h
    confidence DOUBLE PRECISION, -- Confidence level (0-1)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    day_of_week INTEGER, -- 0=Monday, 6=Sunday
    hour INTEGER, -- 0-23
    free_flow_speed DOUBLE PRECISION, -- Free flow speed in km/h
    road_class TEXT, -- Functional Road Class
    source TEXT DEFAULT 'TomTom',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create spatial index for efficient geo-queries
CREATE INDEX IF NOT EXISTS idx_traffic_readings_location ON traffic_readings USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_traffic_readings_timestamp ON traffic_readings(timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_readings_lat_lon ON traffic_readings(lat, lon);
CREATE INDEX IF NOT EXISTS idx_traffic_readings_day_hour ON traffic_readings(day_of_week, hour);

-- Prediction locations table (monitored locations)
CREATE TABLE IF NOT EXISTS prediction_locations (
    id SERIAL PRIMARY KEY,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    location GEOMETRY(POINT, 4326),
    location_name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    priority INTEGER DEFAULT 1, -- Higher priority = more frequent polling
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(lat, lon)
);

-- Create spatial index
CREATE INDEX IF NOT EXISTS idx_prediction_locations_location ON prediction_locations USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_prediction_locations_active ON prediction_locations(is_active, priority);

-- Model predictions table (stores ML predictions)
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    lat DOUBLE PRECISION NOT NULL,
    lon DOUBLE PRECISION NOT NULL,
    location GEOMETRY(POINT, 4326),
    predicted_speed DOUBLE PRECISION NOT NULL, -- Predicted speed in km/h
    predicted_traffic_level TEXT, -- 'low', 'medium', 'high'
    prediction_timestamp TIMESTAMPTZ NOT NULL,
    confidence DOUBLE PRECISION, -- Model confidence (0-1)
    model_version TEXT, -- Model version identifier
    prediction_type TEXT, -- 'current' or 'future_30min'
    minutes_ahead INTEGER, -- For future predictions (0 for current)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_model_predictions_location ON model_predictions USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_model_predictions_timestamp ON model_predictions(prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_model_predictions_type ON model_predictions(prediction_type);

-- Model metadata table (tracks model versions and performance)
CREATE TABLE IF NOT EXISTS model_metadata (
    id SERIAL PRIMARY KEY,
    model_version TEXT UNIQUE NOT NULL,
    model_type TEXT NOT NULL, -- 'current' or 'future_30min'
    file_path TEXT NOT NULL,
    training_date TIMESTAMPTZ NOT NULL,
    mae DOUBLE PRECISION, -- Mean Absolute Error
    rmse DOUBLE PRECISION, -- Root Mean Squared Error
    r2_score DOUBLE PRECISION,
    accuracy DOUBLE PRECISION, -- For classification tasks
    training_samples INTEGER,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- API request logs (for monitoring)
CREATE TABLE IF NOT EXISTS api_logs (
    id SERIAL PRIMARY KEY,
    endpoint TEXT NOT NULL,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    response_time_ms INTEGER,
    status_code INTEGER,
    error_message TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON api_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint);

-- Function to update location geometry from lat/lon
CREATE OR REPLACE FUNCTION update_location_geometry()
RETURNS TRIGGER AS $$
BEGIN
    NEW.location = ST_SetSRID(ST_MakePoint(NEW.lon, NEW.lat), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers to automatically update geometry
CREATE TRIGGER update_traffic_readings_geometry
    BEFORE INSERT OR UPDATE ON traffic_readings
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geometry();

CREATE TRIGGER update_prediction_locations_geometry
    BEFORE INSERT OR UPDATE ON prediction_locations
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geometry();

CREATE TRIGGER update_model_predictions_geometry
    BEFORE INSERT OR UPDATE ON model_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_location_geometry();

