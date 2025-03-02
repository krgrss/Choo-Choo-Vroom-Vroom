import os
import pandas as pd
import numpy as np

def add_time_features(df):
    """
    Extract additional time-based features from 'timestamp':
      - 'hour' (0–23)
      - 'month' (1–12)
      - 'is_weekend' (0 or 1)
      - Optionally keep or transform existing 'day_of_week'
    """
    # Convert day_of_week to numeric if needed. 
    # Currently, 'day_of_week' is a string like "Monday", "Tuesday", etc.
    # If that's the case, we can map them to 0..6 or keep them as is.
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    # Some data might already be numeric or partial, so do a safe map:
    df["day_num"] = df["day_of_week"].map(day_map)
    
    # If the dictionary indicates 'day_of_week' is already numeric (e.g., 0=Monday),
    # you can skip the mapping above.
    
    # Add hour, month, weekend flags
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    # day_num could be used to define weekend
    df["is_weekend"] = df["day_num"].isin([5, 6]).astype(int)
    
    return df


def merge_weather_data(df, weather_path="data/external/weather_hourly.csv"):
    """
    Merge external weather data on an hourly basis if the CSV is available.
    The CSV is expected to have a date/time column (e.g., 'datetime_hour') and relevant weather columns:
      ['datetime_hour', 'temp_C', 'precip_mm', 'snow_cm', etc.]
    We align them by the nearest hour to 'timestamp' in the main DataFrame.
    """
    if not os.path.exists(weather_path):
        print(f"Weather file not found at {weather_path}, skipping weather merge.")
        # If missing, we can fill placeholders or just return df
        return df

    weather_df = pd.read_csv(weather_path, parse_dates=["datetime_hour"])
    # e.g., weather_df columns: ["datetime_hour", "temp_C", "precip_mm", "snow_cm", ...]

    # For alignment, we'll create an hourly key in both dataframes
    df["timestamp_hour"] = df["timestamp"].dt.floor("H")
    weather_df["datetime_hour"] = weather_df["datetime_hour"].dt.floor("H")

    # Merge
    merged = pd.merge(df, weather_df, left_on="timestamp_hour", right_on="datetime_hour", how="left")

    # Drop the helper columns
    merged.drop(["timestamp_hour", "datetime_hour"], axis=1, inplace=True)
    
    return merged


def merge_event_data(df, events_path="data/external/events.csv"):
    """
    Merge data about special events or holidays on a daily basis if the CSV is available.
    Suppose events.csv has columns: [date, event_name, event_type, etc.]
    We'll create a flag 'has_event' = 1 if there's any event on that date, else 0.
    """
    if not os.path.exists(events_path):
        print(f"Events file not found at {events_path}, skipping event merge.")
        df["has_event"] = 0
        return df
    
    events_df = pd.read_csv(events_path, parse_dates=["date"])
    # Possibly multiple events in a single date. We'll reduce to a daily level:
    events_df["date"] = events_df["date"].dt.normalize()  # sets to midnight for daily merges
    events_df["has_event"] = 1
    daily_events = events_df.groupby("date")["has_event"].max().reset_index()

    # Create a daily column from the main df
    df["date_only"] = df["timestamp"].dt.normalize()
    
    merged = pd.merge(df, daily_events, left_on="date_only", right_on="date", how="left")
    merged["has_event"] = merged["has_event"].fillna(0).astype(int)

    merged.drop(["date_only", "date"], axis=1, inplace=True)
    return merged


def feature_engineering_pipeline(df):
    """
    A pipeline that applies all feature engineering steps in sequence:
      1) Time-based feature extraction
      2) Weather data merge
      3) Event data merge
    Returns an enriched DataFrame.
    """
    df = add_time_features(df)
    df = merge_weather_data(df, weather_path="data/external/weather_hourly.csv")
    df = merge_event_data(df, events_path="data/external/events.csv")

    # Example: fill missing weather columns or create default values
    # if they are not present or partially missing:
    for col in ["temp_C", "precip_mm", "snow_cm"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
        else:
            df[col] = 0  # create a column if it doesn't exist

    return df


if __name__ == "__main__":
    # Example usage: Suppose we already loaded the unified data from data_preprocessing.
    # We simulate that step below just for demonstration.

    # In reality, you'd do something like:
    # from data_preprocessing import main_preprocessing
    # combined_df = main_preprocessing()
    # Then pass it to the pipeline:
    # combined_enriched = feature_engineering_pipeline(combined_df)

    # For demonstration, let's create a small dummy DataFrame:
    dummy_data = {
        "timestamp": pd.date_range(start="2024-01-01 06:00", periods=5, freq="H"),
        "day_of_week": ["Monday", "Monday", "Monday", "Monday", "Monday"],
        "mode": ["bus", "bus", "subway", "streetcar", "bus"],
        "location": ["LocA", "LocB", "LocC", "LocD", "LocE"],
        "incident": ["Mechanical", "Security", "EUBK", "Diversion", "Mechanical"],
        "delay_minutes": [10, 5, 3, 20, 15]
    }
    dummy_df = pd.DataFrame(dummy_data)
    
    # Apply pipeline
    enriched_df = feature_engineering_pipeline(dummy_df)
    print("Enriched DataFrame:\n", enriched_df.head(10))
