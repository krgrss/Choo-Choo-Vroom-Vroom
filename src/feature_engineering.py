import os
import pandas as pd
import numpy as np
import requests
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta

#############################
# Weather Data Functions
#############################

def fetch_open_meteo_weather(latitude=43.65, longitude=-79.38, start_date='2024-01-01', end_date='2024-01-07'):
    """
    Fetches hourly historical weather data from the Open-Meteo Archive API for the specified location and date range.
    Returns a DataFrame with hourly weather data, including:
      - datetime_hour: datetime of observation
      - temperature_2m: temperature in Celsius
      - precipitation: precipitation amount
    """
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
        print("No hourly weather data found in the API response.")
        return pd.DataFrame()
    df_weather = pd.DataFrame(hourly_data)
    df_weather["datetime_hour"] = pd.to_datetime(df_weather["time"])
    df_weather.drop(columns=["time"], inplace=True)
    return df_weather

def merge_weather_data(df, start_date="2024-01-01", end_date="2024-01-31", latitude=43.65, longitude=-79.38):
    """
    Merges historical weather data from Open-Meteo Archive into the transit DataFrame.
    Assumes the transit DataFrame has a 'timestamp' column (as datetime).
    Rounds transit timestamps to the nearest hour and merges on that key.
    """
    weather_df = fetch_open_meteo_weather(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)
    if weather_df.empty:
        print("Weather DataFrame is empty; skipping weather merge.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    weather_df["datetime_hour"] = weather_df["datetime_hour"].dt.floor("H")
    merged = pd.merge(df, weather_df, left_on="timestamp_hour", right_on="datetime_hour", how="left")
    merged.drop(columns=["timestamp_hour", "datetime_hour"], inplace=True)
    return merged

#############################
# Time and Lag Features
#############################

def add_time_features(df):
    """
    Adds time-based features:
      - hour: hour of day (0-23)
      - month: month of year (1-12)
      - is_weekend: 1 if Saturday or Sunday, else 0
      - day_num: numeric day of week (0=Monday,...,6=Sunday)
    """
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
    """
    Adds lag features capturing historical delay patterns:
      - rolling_delay_mean: rolling average of 'delay_minutes' over the specified window.
      - rolling_delay_count: rolling count of delay events (where delay_minutes > 0) over the window.
    """
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)
    df["rolling_delay_mean"] = df["delay_minutes"].rolling(window=window).mean()
    df["rolling_delay_count"] = (df["delay_minutes"] > 0).rolling(window=window).sum()
    df.reset_index(inplace=True)
    return df

#############################
# GTFS Route Information Functions
#############################

def load_gtfs_route_mapping(stops_path="data/external/stops.txt",
                              stop_times_path="data/external/stop_times.txt",
                              trips_path="data/external/trips.txt",
                              routes_path="data/external/routes.txt"):
    """
    Loads GTFS files and creates a mapping from normalized stop names to route information.
    Returns a dictionary: { normalized_stop_name: route_info }
    """
    if not os.path.exists(stops_path):
        print(f"Stops file not found at {stops_path}. Returning empty dictionary.")
        return {}
    
    stops_df = pd.read_csv(stops_path)
    stop_times_df = pd.read_csv(stop_times_path) if os.path.exists(stop_times_path) else pd.DataFrame()
    trips_df = pd.read_csv(trips_path) if os.path.exists(trips_path) else pd.DataFrame()
    routes_df = pd.read_csv(routes_path) if os.path.exists(routes_path) else pd.DataFrame()
    
    if stop_times_df.empty or trips_df.empty or routes_df.empty:
        print("One or more GTFS files (stop_times, trips, or routes) are missing or empty.")
        return {}
    
    # Merge stop_times with trips to get route_id for each stop.
    stop_trips = pd.merge(stop_times_df, trips_df, on="trip_id", how="left")
    # Merge with routes to get route_short_name.
    stop_trips = pd.merge(stop_trips, routes_df, on="route_id", how="left")
    # Group by stop_id to get unique route_short_names.
    mapping = stop_trips.groupby("stop_id")["route_short_name"].unique().reset_index()
    mapping["route_info"] = mapping["route_short_name"].apply(lambda x: ", ".join(x.astype(str)))
    mapping = mapping[["stop_id", "route_info"]]
    
    # Merge with stops to get stop_name.
    stops_routes = pd.merge(stops_df, mapping, on="stop_id", how="left")
    stops_routes["norm_stop_name"] = stops_routes["stop_name"].str.lower().str.strip()
    route_mapping = dict(zip(stops_routes["norm_stop_name"], stops_routes["route_info"]))
    return route_mapping

def add_route_info(df, gtfs_path="data/external"):
    """
    Adds route information to the transit DataFrame using GTFS data.
    Assumes df has a 'location' column corresponding to stop names.
    Uses fuzzy matching to map transit locations to GTFS stops.
    """
    stops_path = os.path.join(gtfs_path, "stops.txt")
    stop_times_path = os.path.join(gtfs_path, "stop_times.txt")
    trips_path = os.path.join(gtfs_path, "trips.txt")
    routes_path = os.path.join(gtfs_path, "routes.txt")
    
    # Load GTFS mapping.
    route_mapping = load_gtfs_route_mapping(stops_path, stop_times_path, trips_path, routes_path)
    
    # Normalize the transit location.
    df["norm_location"] = df["location"].str.lower().str.strip()
    
    def map_stop(norm_loc):
        # Direct match first.
        if norm_loc in route_mapping:
            return route_mapping[norm_loc]
        # Use fuzzy matching as a fallback.
        match = process.extractOne(norm_loc, list(route_mapping.keys()), scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 80:  # threshold can be adjusted.
            return route_mapping[match[0]]
        else:
            return None

    df["route"] = df["norm_location"].apply(map_stop)
    df.drop(columns=["norm_location"], inplace=True)
    return df

#############################
# Feature Engineering Pipeline
#############################

def feature_engineering_pipeline(df):
    """
    Applies the following feature engineering steps:
      1. Ensure 'timestamp' is in datetime format.
      2. Add basic time-based features.
      3. Merge historical weather data from Open-Meteo Archive.
      4. Add lag features (rolling delay metrics).
      5. Add route information from GTFS files.
      6. Fill missing weather data with defaults.
    Returns the enriched DataFrame.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_time_features(df)
    
    # Set weather data date range (adjust as needed)
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    df = merge_weather_data(df, start_date=start_date, end_date=end_date, latitude=43.65, longitude=-79.38)
    
    df = add_lag_features(df, window="1H")
    
    # Add route information using GTFS .txt files.
    df = add_route_info(df, gtfs_path="data/external")
    
    # Fill missing weather data with defaults.
    for col in ["temperature_2m", "precipitation"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0
    
    return df

if __name__ == "__main__":
    # Load your real raw transit data from disk.
    # For example, if you have a CSV file "raw_transit_data.csv" containing the combined data:
    try:
        raw_df = pd.read_csv("data/raw_transit_data.csv", parse_dates=["timestamp"])
        print("Raw transit data loaded. Shape:", raw_df.shape)
    except Exception as e:
        print("Error loading raw transit data:", e)
        exit()

    # Process the raw data with the feature engineering pipeline.
    enriched_df = feature_engineering_pipeline(raw_df)
    print("Enriched DataFrame (first 10 rows):\n", enriched_df.head(10))
    
    # Save the enriched data for further use.
    enriched_df.to_csv("enriched_data.csv", index=False)
    print("Enriched data saved to 'enriched_data.csv'.")
