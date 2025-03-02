import pandas as pd
import numpy as np
import os

def load_subway_codes(codes_path="data/subway-delay-codes.csv"):
    """
    Loads subway delay codes into a dictionary: 
    { SUB RMENU CODE -> CODE DESCRIPTION }
    Example columns: [SUB RMENU CODE, CODE DESCRIPTION]
    """
    codes_df = pd.read_csv(codes_path)
    code_map = dict(zip(codes_df["SUB RMENU CODE"], codes_df["CODE DESCRIPTION"]))
    return code_map

def load_stops(stops_path="data/external/gtfs/stops.txt"):
    """
    Loads GTFS stops.txt, presumed comma-delimited.
    Expects columns: stop_id, stop_name, stop_lat, stop_lon, etc.
    Returns a DataFrame with columns [stop_name, stop_lat, stop_lon, norm_stop_name].
    """
    if not os.path.exists(stops_path):
        print(f"Stops file not found at {stops_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["stop_name", "stop_lat", "stop_lon", "norm_stop_name"])
    
    stops_df = pd.read_csv(stops_path, sep=",")  # or sep="\t" if needed
    needed = ["stop_name", "stop_lat", "stop_lon"]
    for col in needed:
        if col not in stops_df.columns:
            print(f"Column '{col}' not found in stops.txt, continuing with partial data.")
    
    stops_df = stops_df[[c for c in needed if c in stops_df.columns]].copy()
    if "stop_lat" in stops_df.columns:
        stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    if "stop_lon" in stops_df.columns:
        stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")

    stops_df["norm_stop_name"] = stops_df["stop_name"].astype(str).str.lower().str.strip()
    return stops_df

def load_bus_data(bus_path="data/bus-data.csv"):
    df_bus = pd.read_csv(bus_path, dtype=str)
    df_bus["datetime_str"] = df_bus["Date"].str.strip() + " " + df_bus["Time"].str.strip()
    df_bus["timestamp"] = pd.to_datetime(
        df_bus["datetime_str"],
        format="%d-%b-%y %H:%M",       # NEW for "1-Jan-24 02:08"
        errors="coerce"
    )

    rename_map = {
        "Date": "date",
        "Time": "time",
        "Route": "route",
        "Day": "day_of_week",
        "Location": "location",
        "Incident": "incident",
        "Min Delay": "delay_minutes",
        "Min Gap": "gap_minutes",
        "Direction": "direction",
        "Vehicle": "vehicle"
    }
    df_bus.rename(columns=rename_map, inplace=True)
    df_bus["delay_minutes"] = pd.to_numeric(df_bus["delay_minutes"], errors="coerce")
    df_bus["gap_minutes"] = pd.to_numeric(df_bus["gap_minutes"], errors="coerce")
    df_bus["mode"] = "bus"
    return df_bus

def load_streetcar_data(streetcar_path="data/streetcar-data.csv"):
    df_stc = pd.read_csv(streetcar_path, dtype=str)
    df_stc["datetime_str"] = df_stc["Date"].str.strip() + " " + df_stc["Time"].str.strip()
    df_stc["timestamp"] = pd.to_datetime(
        df_stc["datetime_str"],
        format="%d-%b-%y %H:%M",        # NEW for "1-Jan-24 02:45"
        errors="coerce"
    )

    rename_map = {
        "Date": "date",
        "Time": "time",
        "Route": "route",
        "Day": "day_of_week",
        "Location": "location",
        "Incident": "incident",
        "Min Delay": "delay_minutes",
        "Min Gap": "gap_minutes",
        "Direction": "direction",
        "Vehicle": "vehicle"
    }
    df_stc.rename(columns=rename_map, inplace=True)
    df_stc["delay_minutes"] = pd.to_numeric(df_stc["delay_minutes"], errors="coerce")
    df_stc["gap_minutes"] = pd.to_numeric(df_stc["gap_minutes"], errors="coerce")
    df_stc["mode"] = "streetcar"
    return df_stc

def load_subway_data(subway_path="data/subway-data.csv", codes_map=None):
    df_sub = pd.read_csv(subway_path, dtype=str)
    df_sub["datetime_str"] = df_sub["Date"].str.strip() + " " + df_sub["Time"].str.strip()
    df_sub["timestamp"] = pd.to_datetime(
        df_sub["datetime_str"],
        format="%Y/%m/%d %H:%M",   # for "2024/01/01 02:00"
        errors="coerce"
    )

    rename_map = {
        "Date": "date",
        "Time": "time",
        "Day": "day_of_week",
        "Station": "location",
        "Code": "incident_code",
        "Min Delay": "delay_minutes",
        "Min Gap": "gap_minutes",
        "Bound": "direction",
        "Line": "line",
        "Vehicle": "vehicle"
    }
    df_sub.rename(columns=rename_map, inplace=True)
    df_sub["delay_minutes"] = pd.to_numeric(df_sub["delay_minutes"], errors="coerce")
    df_sub["gap_minutes"] = pd.to_numeric(df_sub["gap_minutes"], errors="coerce")
    df_sub["mode"] = "subway"

    if codes_map is not None:
        df_sub["incident"] = df_sub["incident_code"].map(codes_map)
        df_sub["incident"] = df_sub["incident"].fillna(df_sub["incident_code"])
    else:
        df_sub["incident"] = df_sub["incident_code"]
    
    return df_sub

def unify_datasets(bus_df, streetcar_df, subway_df):
    bus_df["line"] = pd.NA
    streetcar_df["line"] = pd.NA
    subway_df["route"] = pd.NA

    common_cols = [
        "timestamp", "mode", "route", "line", "location", "incident",
        "delay_minutes", "gap_minutes", "direction", "day_of_week", "vehicle"
    ]
    bus_df = bus_df.reindex(columns=common_cols)
    streetcar_df = streetcar_df.reindex(columns=common_cols)
    subway_df = subway_df.reindex(columns=common_cols)

    combined = pd.concat([bus_df, streetcar_df, subway_df], ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

def merge_stops_info(combined_df, stops_df):
    """
    Merges lat/lon from stops_df into combined_df by direct string match:
    norm(location) -> norm(stop_name).
    If no match is found, lat/lon remain NaN.
    """
    if stops_df.empty:
        print("Stops DataFrame is empty; skipping stops merge.")
        return combined_df

    combined_df["norm_location"] = combined_df["location"].astype(str).str.lower().str.strip()
    
    merged = pd.merge(
        combined_df,
        stops_df[["norm_stop_name", "stop_lat", "stop_lon"]],
        left_on="norm_location",
        right_on="norm_stop_name",
        how="left"
    )
    merged.drop(columns=["norm_stop_name"], inplace=True, errors="ignore")
    merged.drop(columns=["norm_location"], inplace=True, errors="ignore")

    return merged

def main_preprocessing(
    bus_path="data/bus-data.csv",
    streetcar_path="data/streetcar-data.csv",
    subway_path="data/subway-data.csv",
    codes_path="data/subway-delay-codes.csv",
    stops_path="data/external/gtfs/stops.txt"
):
    """
    1) Load bus, streetcar, subway data (optionally with codes for subway).
    2) Unify them in one DataFrame with consistent columns.
    3) Load stops.txt and merge lat/lon into the dataset by direct string match.
    4) Basic cleaning of invalid rows.
    5) Return the combined DataFrame.
    """
    codes_map = load_subway_codes(codes_path)
    df_bus = load_bus_data(bus_path)
    df_stc = load_streetcar_data(streetcar_path)
    df_sub = load_subway_data(subway_path, codes_map=codes_map)

    combined_df = unify_datasets(df_bus, df_stc, df_sub)
    
    # Clean invalid or missing data
    combined_df.dropna(subset=["timestamp", "delay_minutes"], inplace=True)
    combined_df = combined_df[combined_df["delay_minutes"] >= 0]

    # Merge stops data
    stops_df = load_stops(stops_path=stops_path)
    combined_df = merge_stops_info(combined_df, stops_df)

    return combined_df

if __name__ == "__main__":
    combined = main_preprocessing()
    print("Combined dataset shape:", combined.shape)
    print(combined.head(30))
