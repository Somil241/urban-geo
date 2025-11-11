import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os 
import numpy as np

# Load environment variables
load_dotenv()
engine = create_engine(os.getenv("POSTGRES_URL"))

def increment_lat_lon(lat, lon, delta=0.01):
    """Increment latitude and longitude to simulate route traversal"""
    return lat + delta, lon + delta

def analyze_route(start_lat, start_lon, end_lat, end_lon, delta=0.01):
    """
    Fetch and analyze average speed and delay for route segments between two coordinates.
    The function steps through the route using increment_lat_lon().
    """
    lat, lon = start_lat, start_lon
    route_data = []

    # Generate intermediate points between start and end
    while lat <= end_lat and lon <= end_lon:
        query = f"""
            SELECT * FROM route_traffic_data
            WHERE start_lat={lat:.4f} AND start_lon={lon:.4f}
        """
        df = pd.read_sql(query, engine, parse_dates=["timestamp"])

        if not df.empty:
            df["hour"] = df["timestamp"].dt.hour
            hourly_stats = df.groupby("hour").agg(
                avg_speed=("avg_speed_kmph", "mean"),
                avg_delay=("traffic_delay_sec", "mean"),
                avg_time=("travel_time_sec", "mean"),
                count=("timestamp", "count")
            ).reset_index()

            route_data.append(hourly_stats)

        # Move to the next coordinate along the route
        lat, lon = increment_lat_lon(lat, lon, delta)

    if not route_data:
        print("No traffic data found along this route.")
        return

    # Combine all hourly dataframes
    combined_df = pd.concat(route_data)
    final_stats = combined_df.groupby("hour").mean().reset_index()

    print("\n=== Traffic Analysis for Route ===")
    print(final_stats)

    avg_speed = final_stats["avg_speed"].mean()
    avg_delay = final_stats["avg_delay"].mean()

    print(f"\nOverall average speed: {avg_speed:.2f} km/h")
    print(f"Overall average delay: {avg_delay:.2f} sec")

if __name__ == "__main__":
    # Example: Connaught Place → IGI Airport
    analyze_route(28.6315, 77.2167, 28.5562, 77.1000, delta=0.01)
