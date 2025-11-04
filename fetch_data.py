import os
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")

# -----------------------------
# Database Connection
# -----------------------------
def get_db_engine():
    return create_engine(POSTGRES_URL)

# -----------------------------
# Create Table if not exists
# -----------------------------
def ensure_table_exists():
    engine = get_db_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMPTZ NOT NULL,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                current_speed DOUBLE PRECISION,
                free_flow_speed DOUBLE PRECISION,
                confidence DOUBLE PRECISION,
                road_class TEXT,
                source TEXT
            );
        """))
    print("🧱 Ensured 'traffic_data' table exists in Neon.")

# -----------------------------
# Fetch Data from TomTom API
# -----------------------------
def fetch_tomtom_data(lat, lon):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{lat},{lon}"
    }

    retries, backoff = 0, 1
    while retries < 5:
        try:
            res = requests.get(url, params=params, timeout=10)
            if res.status_code == 200:
                return res.json()
            else:
                print(f"Error {res.status_code}, retrying in {backoff}s...")
        except Exception as e:
            print(f"Error fetching data: {e}")

        time.sleep(backoff)
        retries += 1
        backoff *= 2

    print("❌ Failed to fetch data after multiple retries.")
    return None

# -----------------------------
# Process Data
# -----------------------------
def process_data(api_data, lat, lon):
    if not api_data or "flowSegmentData" not in api_data:
        print("No valid data returned.")
        return None

    segment = api_data["flowSegmentData"]
    df = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc),
        "latitude": lat,
        "longitude": lon,
        "current_speed": segment.get("currentSpeed"),
        "free_flow_speed": segment.get("freeFlowSpeed"),
        "confidence": segment.get("confidence"),
        "road_class": segment.get("frc"),
        "source": "TomTom"
    }])
    return df

# -----------------------------
# Save to Neon PostgreSQL
# -----------------------------
def save_to_postgres(df):
    if df is None or df.empty:
        print("⚠️ No data to save.")
        return

    engine = get_db_engine()
    df.to_sql("traffic_data", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df)} record(s) to Neon PostgreSQL.")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    lat, lon = 28.6139, 75.209 # Example: Delhi, India
    print(f"Fetching TomTom traffic data for ({lat}, {lon})...")

    ensure_table_exists()

    api_data = fetch_tomtom_data(lat, lon)
    df = process_data(api_data, lat, lon)
    save_to_postgres(df)
