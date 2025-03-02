import os
import pandas as pd
import numpy as np
import requests
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta

def fetch_open_meteo_weather(latitude=43.65, longitude=-79.38, start_date='2024-01-01', end_date='2024-12-31'):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation"
        f"&timezone=auto"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching weather data:", response.status_code, response.text)
        return pd.DataFrame()
    data = response.json()
    hourly_data = data.get("hourly", {})
    if not hourly_data:
        print("No hourly weather data found in API response.")
        return pd.DataFrame()
    df_weather = pd.DataFrame(hourly_data)
    df_weather["datetime_hour"] = pd.to_datetime(df_weather["time"])
    df_weather.drop(columns=["time"], inplace=True)
    return df_weather

def merge_weather_data(df, start_date="2024-01-01", end_date="2024-12-31", latitude=43.65, longitude=-79.38):
    weather_df = fetch_open_meteo_weather(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)
    if weather_df.empty:
        print("Weather DataFrame empty; skipping weather merge.")
        return df
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    weather_df["datetime_hour"] = weather_df["datetime_hour"].dt.floor("H")
    merged = pd.merge(df, weather_df, left_on="timestamp_hour", right_on="datetime_hour", how="left")
    merged.drop(columns=["timestamp_hour", "datetime_hour"], inplace=True)
    return merged

def add_time_features(df):
    if "day_of_week" in df.columns:
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}
        df["day_num"] = df["day_of_week"].map(day_map)
    else:
        df["day_num"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_num"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, window="1H"):
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    df["rolling_delay_mean"] = df["delay_minutes"].rolling(window=window).mean()
    df["rolling_delay_count"] = (df["delay_minutes"] > 0).rolling(window=window).sum()
    df.reset_index(inplace=True)
    return df

def load_gtfs_route_mapping(stops_path="data/external/gtfs/stops.txt",
                            stop_times_path="data/external/gtfs/stop_times.txt",
                            trips_path="data/external/gtfs/trips.txt",
                            routes_path="data/external/gtfs/routes.txt"):
    """
    If you have GTFS data, read from these .txt files, build a dict norm_stop_name -> route_info
    For now, a placeholder. Adjust if you have actual GTFS merges.
    """
    if not os.path.exists(stops_path):
        print(f"Stops file not found at {stops_path}. Return empty dict.")
        return {}

    # Minimal approach
    route_mapping = {}
    # In a real scenario, you'd merge stop_times, trips, routes, etc.
    return route_mapping

def add_route_info(df, gtfs_path="data/external/gtfs"):
    stops_path = os.path.join(gtfs_path, "stops.txt")
    stop_times_path = os.path.join(gtfs_path, "stop_times.txt")
    trips_path = os.path.join(gtfs_path, "trips.txt")
    routes_path = os.path.join(gtfs_path, "routes.txt")

    route_mapping = load_gtfs_route_mapping(stops_path, stop_times_path, trips_path, routes_path)

    df["norm_location"] = df["location"].astype(str).str.lower().str.strip()

    def map_stop(norm_loc):
        if not route_mapping:
            return None
        if norm_loc in route_mapping:
            return route_mapping[norm_loc]
        match = process.extractOne(norm_loc, list(route_mapping.keys()), scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 80:
            return route_mapping[match[0]]
        return None

    df["route"] = df["norm_location"].apply(map_stop)
    df.drop(columns=["norm_location"], inplace=True)
    return df

def feature_engineering_pipeline(df):
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df = add_time_features(df)
    df = merge_weather_data(df, start_date="2024-01-01", end_date="2024-12-31")
    df = add_lag_features(df, window="1H")
    df = add_route_info(df, gtfs_path="data/external/gtfs")

    # Fill missing weather columns
    for col in ["temperature_2m", "precipitation"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    return df

if __name__ == "__main__":
    print("feature_engineering.py - for local testing only.")
