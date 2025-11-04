import os
import requests
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY")
DB_URL = os.getenv("POSTGRES_URL")
engine = create_engine(DB_URL)

def fetch_route_data(start_lat, start_lon, end_lat, end_lon):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{start_lat},{start_lon}:{end_lat},{end_lon}/json"
    params = {
        "key": TOMTOM_API_KEY,
        "traffic": "true",
    }

    res = requests.get(url, params=params)
    data = res.json()

    if "routes" not in data:
        print("❌ Error fetching route data:", data)
        return None

    summary = data["routes"][0]["summary"]
    travel_time = summary["travelTimeInSeconds"]
    traffic_delay = summary.get("trafficDelayInSeconds", 0)
    length = summary["lengthInMeters"]
    average_speed = (length / travel_time) * 3.6  # m/s → km/h

    df = pd.DataFrame([{
        "timestamp": datetime.utcnow(),
        "start_lat": start_lat,
        "start_lon": start_lon,
        "end_lat": end_lat,
        "end_lon": end_lon,
        "travel_time_sec": travel_time,
        "traffic_delay_sec": traffic_delay,
        "length_m": length,
        "avg_speed_kmph": average_speed,
        "source": "TomTom"
    }])

    return df

def save_to_db(df):
    df.to_sql("route_traffic_data", engine, if_exists="append", index=False)
    print(f"✅ Saved {len(df)} route record(s) to PostgreSQL.")

if __name__ == "__main__":
    # Example: Connaught Place to IGI Airport
    start_lat, start_lon = 28.6315, 77.2167
    end_lat, end_lon = 28.5562, 77.1000

    print("Fetching route traffic data...")
    df = fetch_route_data(start_lat, start_lon, end_lat, end_lon)
    if df is not None:
        save_to_db(df)
