import pandas as pd
import numpy as np
import os

def load_subway_codes(codes_path="data/subway-delay-codes.csv"):
    """
    Loads subway delay codes into a DataFrame with 'Code' and 'Code Description' columns.
    Returns a dictionary mapping code -> description, for easy lookup.
    """
    codes_df = pd.read_csv(codes_path)
    # Example structure:
    #   Code | Code Description
    #   EUBK | Brakes
    #   EUAC | Air Conditioning
    # etc.

    # Create a dictionary: {"EUBK": "Brakes", ...}
    code_map = dict(zip(codes_df["SUB RMENU CODE"], codes_df["CODE DESCRIPTION"]))
    return code_map


def load_bus_data(bus_path="data/bus-data.csv"):
    """
    Loads the bus data according to the data dictionary:
      - Date: DD/MM/YYYY
      - Time: hh:mm:ss AM/PM
      - Route, Day, Location, Incident, Min Delay, Min Gap, Direction, Vehicle
    Parses date/time into a single datetime column: 'timestamp'.
    """
    df_bus = pd.read_csv(bus_path, dtype=str)  # read as strings first

    # Combine Date + Time into a single string for parsing.
    df_bus["datetime_str"] = df_bus["Date"].str.strip() + " " + df_bus["Time"].str.strip()

    # Parse to datetime.
    # Format example: "20/06/2017 12:35:00 AM" => dayfirst=True, 12-hour clock with %I:%M:%S %p
    df_bus["timestamp"] = pd.to_datetime(
        df_bus["datetime_str"],
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Rename columns to be consistent with the dictionary & subsequent usage
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

    # Convert numeric fields
    df_bus["delay_minutes"] = pd.to_numeric(df_bus["delay_minutes"], errors="coerce")
    df_bus["gap_minutes"] = pd.to_numeric(df_bus["gap_minutes"], errors="coerce")

    # Insert a 'mode' column for clarity
    df_bus["mode"] = "bus"

    return df_bus


def load_streetcar_data(streetcar_path="data/streetcar-data.csv"):
    """
    Streetcar data matches the bus dictionary fields:
      - Date (DD/MM/YYYY), Time (hh:mm:ss AM/PM), Route, Day, Location, Incident,
        Min Delay, Min Gap, Direction, Vehicle
    The parsing is essentially the same as bus data.
    """
    df_stc = pd.read_csv(streetcar_path, dtype=str)

    df_stc["datetime_str"] = df_stc["Date"].str.strip() + " " + df_stc["Time"].str.strip()
    df_stc["timestamp"] = pd.to_datetime(
        df_stc["datetime_str"],
        format="%d/%m/%Y %I:%M:%S %p",
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
    """
    Loads subway data according to the dictionary:
      - Date (DD/MM/YYYY), Time (24h), Day, Station, Code, Min Delay, Min Gap, Bound, Line, Vehicle
    Then merges with the codes_map (dict) to get descriptive 'incident' text from the Code.
    """
    df_sub = pd.read_csv(subway_path, dtype=str)

    # Combine Date + Time (which is a 24h string like "1:59") into one. 
    # The dictionary says "31/12/2016 1:59", so let's parse dayfirst.
    df_sub["datetime_str"] = df_sub["Date"].str.strip() + " " + df_sub["Time"].str.strip()
    df_sub["timestamp"] = pd.to_datetime(
        df_sub["datetime_str"],
        format="%d/%m/%Y %H:%M",
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

    # Convert numeric
    df_sub["delay_minutes"] = pd.to_numeric(df_sub["delay_minutes"], errors="coerce")
    df_sub["gap_minutes"] = pd.to_numeric(df_sub["gap_minutes"], errors="coerce")

    df_sub["mode"] = "subway"

    # Map subway "Code" to an incident description if we have a codes_map
    if codes_map is not None:
        # Create a new column "incident" from the code map
        df_sub["incident"] = df_sub["incident_code"].map(codes_map)
        # Where no match found, fill with the original code or "Unknown"
        df_sub["incident"] = df_sub["incident"].fillna(df_sub["incident_code"])

    return df_sub


def load_stops(stops_path="data/external/stops.csv"):
    """
    Loads the stops dataset, which contains:
      stop_id, stop_code, stop_name, stop_desc, stop_lat, stop_lon, ...
    Returns a DataFrame with at least columns: [stop_name, stop_lat, stop_lon].
    """
    if not os.path.exists(stops_path):
        print(f"Stops file not found at {stops_path}. Returning empty DataFrame.")
        return pd.DataFrame(columns=["stop_name", "stop_lat", "stop_lon"])

    stops_df = pd.read_csv(stops_path)
    # Keep only the columns we need for merging:
    stops_df = stops_df[["stop_name", "stop_lat", "stop_lon"]].copy()

    # Convert lat/lon to numeric (in case they're read as strings)
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")

    return stops_df


def unify_datasets(bus_df, streetcar_df, subway_df):
    """
    Combines bus, streetcar, and subway data into a single DataFrame with consistent columns.
    Example unified columns: 
      [timestamp, mode, route, line, location, incident, delay_minutes, gap_minutes, 
       direction, day_of_week, vehicle, ...].
    For bus/streetcar, line is typically NaN. For subway, route is typically NaN.
    """
    # For bus/streetcar, we have 'route' but not 'line'
    bus_df["line"] = pd.NA  # not applicable
    streetcar_df["line"] = pd.NA

    # For subway, we typically have 'line' but not 'route'
    subway_df["route"] = pd.NA

    # Ensure all have the same columns:
    common_cols = [
        "timestamp", "mode", "route", "line", "location", "incident",
        "delay_minutes", "gap_minutes", "direction", "day_of_week", "vehicle"
    ]
    
    bus_df = bus_df.reindex(columns=common_cols)
    streetcar_df = streetcar_df.reindex(columns=common_cols)
    subway_df = subway_df.reindex(columns=common_cols)

    # Concatenate
    combined = pd.concat([bus_df, streetcar_df, subway_df], ignore_index=True)
    combined.sort_values("timestamp", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def main_preprocessing(
    bus_path="data/bus-data.csv",
    streetcar_path="data/streetcar-data.csv",
    subway_path="data/subway-data.csv",
    codes_path="data/subway-delay-codes.csv",
    stops_path="data/external/stops.csv"
):
    """
    Main function to read all data sources and return a single combined DataFrame
    that aligns with the Transportation Data Dictionary.
    """
    # 1) Load the subway code map
    codes_map = load_subway_codes(codes_path)
    stops_df = load_stops(stops_path=stops_path)

    # 2) Load each dataset
    df_bus = load_bus_data(bus_path)
    df_stc = load_streetcar_data(streetcar_path)
    df_sub = load_subway_data(subway_path, codes_map=codes_map)

    # 3) Combine into a single DataFrame
    combined_df = unify_datasets(df_bus, df_stc, df_sub)

    # 4) Basic cleaning:
    # Drop rows where timestamp or delay_minutes is missing, etc.
    combined_df.dropna(subset=["timestamp", "delay_minutes"], inplace=True)
    # Also remove negative or impossible delays
    combined_df = combined_df[combined_df["delay_minutes"] >= 0]

    return combined_df


if __name__ == "__main__":
    # Example usage
    combined = main_preprocessing()
    print("Combined dataset shape:", combined.shape)
    print(combined.head(10))
