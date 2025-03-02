import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster

#####################################
# 1) Risk Map with XGBoost Probability
#####################################

def predict_risk_for_map(df, clf_model, feature_cols=None, lat_col="stop_lat", lon_col="stop_lon"):
    """
    Takes an enriched DataFrame (df) that has:
      - hour, month, is_weekend, mode, route_or_line (by default)
      - lat/lon columns (default: 'stop_lat', 'stop_lon')
    and a trained XGBoost classifier (clf_model) that can do predict_proba().
    
    1) Label-encodes 'mode' and 'route_or_line' to match training (warning: may differ if not the same dataset).
    2) Builds an X matrix of your classification features.
    3) Gets predicted probability of delay (class=1).
    4) Merges lat/lon + predicted_risk.

    Returns a DataFrame with columns: [lat, lon, predicted_risk, location, etc.].
    """

    # If the user hasn't specified which features, we assume these five
    if feature_cols is None:
        feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]

    # Check columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("Risk map: missing feature columns:", missing)
        return pd.DataFrame()

    # Ensure lat/lon exist
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"No lat/lon columns found: {lat_col}, {lon_col} in df.")
        return pd.DataFrame()

    # Make a copy
    df_map = df.copy()

    # Label-encode 'mode' and 'route_or_line'
    from sklearn.preprocessing import LabelEncoder
    le_mode = LabelEncoder()
    le_route = LabelEncoder()

    df_map["mode_enc"] = le_mode.fit_transform(df_map["mode"].astype(str))
    df_map["route_or_line_enc"] = le_route.fit_transform(df_map["route_or_line"].astype(str))

    # Build the classification matrix
    X_map = pd.DataFrame({
        "hour": df_map["hour"],
        "month": df_map["month"],
        "is_weekend": df_map["is_weekend"],
        "mode_enc": df_map["mode_enc"],
        "route_or_line_enc": df_map["route_or_line_enc"]
    })

    # Probability of delay
    if hasattr(clf_model, "predict_proba"):
        y_probs = clf_model.predict_proba(X_map)[:, 1]
        df_map["predicted_risk"] = y_probs
    else:
        print("Classifier does not have predict_proba. Aborting.")
        return pd.DataFrame()

    # Create lat/lon columns
    df_map["lat"] = df_map[lat_col]
    df_map["lon"] = df_map[lon_col]

    # Drop missing lat/lon
    df_map.dropna(subset=["lat", "lon", "predicted_risk"], inplace=True)
    return df_map

def plot_risk_map(risk_df, location_col="location", output_html="risk_map.html"):
    """
    Creates a Folium map from 'risk_df', which is the output of 'predict_risk_for_map'.
    Requires columns: [lat, lon, predicted_risk].
    We color-code markers by risk (green < 0.33 < orange < 0.66 < red).
    """

    if risk_df.empty:
        print("Risk DF is empty, cannot plot.")
        return None

    # Center on Toronto
    center = [43.6532, -79.3832]
    m = folium.Map(location=center, zoom_start=12)
    marker_cluster = MarkerCluster().add_to(m)

    def color_for_risk(r):
        if r < 0.33:
            return "green"
        elif r < 0.66:
            return "orange"
        else:
            return "red"

    for idx, row in risk_df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        risk = row["predicted_risk"]
        loc_desc = str(row.get(location_col, f"loc_{idx}"))
        pop_text = f"<b>{loc_desc}</b><br>Risk: {risk:.2f}"

        folium.Marker(
            location=[lat, lon],
            popup=pop_text,
            icon=folium.Icon(color=color_for_risk(risk))
        ).add_to(marker_cluster)

    m.save(output_html)
    print(f"Risk map saved to '{output_html}'.")
    return m

#####################################
# 2) HeatMap of Historical Incidents
#####################################

def create_heatmap_with_markers(
    df,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="heatmap.html"
):
    """
    Aggregates incidents by location, producing a HeatMap weighted by 'incidents'
    and circle markers for each location. 
    'df' should have columns: [lat_col, lon_col, location_col].
    We'll do a groupby on location_col to count how many rows (incidents) each location has.

    1) Group by location_col, get lat/lon from the first row, and 'incidents' = size.
    2) Build a heat_data list for HeatMap.
    3) Add circle markers color-coded by top 25% or not.
    """

    # Drop missing lat/lon
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        print("No data after dropping missing lat/lon.")
        return None

    # Group by location
    grouped = df.groupby(location_col).agg({
        lat_col: "first",
        lon_col: "first",
        location_col: "size"
    }).rename(columns={location_col: "incidents"}).reset_index()

    # Build base map
    m = folium.Map(location=[43.7, -79.4], zoom_start=11, tiles='cartodbpositron')

    # Prepare heat_data
    heat_data = []
    for _, row in grouped.iterrows():
        heat_data.append([
            row[lat_col],
            row[lon_col],
            row["incidents"]
        ])

    HeatMap(
        data=heat_data,
        radius=15,
        min_opacity=0.4,
        blur=10,
        max_zoom=1
    ).add_to(m)

    hi_threshold = grouped["incidents"].quantile(0.75)
    for _, row in grouped.iterrows():
        lat, lon = row[lat_col], row[lon_col]
        inc_count = row["incidents"]
        loc_name = row[location_col]

        color = "red" if inc_count > hi_threshold else "blue"
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            popup=f"{loc_name}<br>Incidents: {inc_count}",
            color='black',
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_html)
    print(f"Heatmap + markers saved to '{output_html}'.")
    return m

#####################################
# 3) Using Subway lat/lon from Excel
#####################################

def create_subway_heatmap_from_excel(
    incidents_csv="data/subway-data.csv",
    latlon_xlsx="data/subway_latlon.xlsx",
    output_html="subway_heatmap.html"
):
    """
    Similar to create_heatmap_with_markers, but merges station lat/lon from an Excel file first.
    Then builds a heatmap & markers.
    """

    # Load incidents
    df_incidents = pd.read_csv(incidents_csv, dtype=str)
    if df_incidents.empty:
        print(f"No data in {incidents_csv}")
        return None

    # e.g. rename 'Station' -> 'station' if needed
    df_incidents.rename(columns={"Station": "station"}, inplace=True, errors="ignore")
    # Convert 'station' to str
    df_incidents["station"] = df_incidents["station"].astype(str).str.strip()

    # Load station lat/lon from Excel
    df_latlon = pd.read_excel(latlon_xlsx)
    df_latlon.rename(columns={
        "Station": "station",
        "Latitude": "lat",
        "Longitude": "lon"
    }, inplace=True, errors="ignore")
    # Convert to float
    df_latlon["lat"] = pd.to_numeric(df_latlon["lat"], errors="coerce")
    df_latlon["lon"] = pd.to_numeric(df_latlon["lon"], errors="coerce")
    df_latlon["station"] = df_latlon["station"].astype(str).str.strip()

    # Merge
    df_merged = pd.merge(
        df_incidents,
        df_latlon[["station", "lat", "lon"]],
        on="station",
        how="left"
    )
    # Drop rows missing lat/lon
    df_merged.dropna(subset=["lat", "lon"], inplace=True)
    if df_merged.empty:
        print("No station-lat/lon matches found.")
        return None

    # Now do the groupby approach
    grouped = df_merged.groupby("station").agg({
        "lat": "first",
        "lon": "first",
        "station": "size"
    }).rename(columns={"station": "incidents"}).reset_index()

    m = folium.Map(location=[43.7, -79.4], zoom_start=11, tiles='cartodbpositron')
    heat_data = [
        [row["lat"], row["lon"], row["incidents"]]
        for _, row in grouped.iterrows()
    ]
    HeatMap(
        data=heat_data,
        radius=15,
        min_opacity=0.4,
        blur=10,
        max_zoom=1
    ).add_to(m)

    hi_threshold = grouped["incidents"].quantile(0.75)
    for _, row in grouped.iterrows():
        color = "red" if row["incidents"] > hi_threshold else "blue"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            popup=f"{row['station']}<br>Incidents: {row['incidents']}",
            color='black',
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    m.save(output_html)
    print(f"Subway Heatmap from Excel saved to '{output_html}'.")
    return m

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_actual_vs_predicted(
    df,
    actual_col="delay_minutes",
    predicted_col="predicted_delay"
):
    """
    Creates a scatter plot of Actual vs. Predicted delay.
    Insight:
      - Points along the diagonal indicate close agreement.
      - Points far from the diagonal show where the prediction was off.
    """
    # Filter rows with valid numeric data
    data = df[[actual_col, predicted_col]].dropna()
    if data.empty:
        print(f"No valid rows to plot for {actual_col} vs {predicted_col}.")
        return

    actual = data[actual_col]
    predicted = data[predicted_col]

    plt.figure()
    plt.scatter(actual, predicted, alpha=0.5)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    # Diagonal line for reference
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel("Actual Delay")
    plt.ylabel("Predicted Delay")
    plt.title("Actual vs Predicted Delay")
    plt.grid(True)
    plt.show()


def plot_most_delay_by_mode(df, mode_col="mode", delay_col="delay_minutes"):
    """
    Shows a bar chart of total delay (or average delay) by each transit mode.
    Insight:
      - Quickly identify which mode (bus, subway, streetcar, etc.)
        tends to accumulate the most total delay.
    """
    # Choose sum or mean of delay based on your preference:
    grouped = df.dropna(subset=[mode_col, delay_col]).groupby(mode_col)[delay_col].sum()

    if grouped.empty:
        print("No valid data to group by mode for delay.")
        return

    # Sort descending for clarity
    grouped = grouped.sort_values(ascending=False)

    plt.figure()
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.xlabel("Mode")
    plt.ylabel("Total Delay (minutes)")
    plt.title("Most Delay by Mode")
    plt.grid(True)
    plt.show()


def plot_place_most_delay(df, location_col="location", delay_col="delay_minutes", top_n=10):
    """
    Bar chart showing which places (locations) have the highest total delay.
    Insight:
      - Useful for spotting the worst-performing stops/stations.
      - top_n defines how many top locations to display.
    """
    group = df.dropna(subset=[location_col, delay_col]).groupby(location_col)[delay_col].sum()
    if group.empty:
        print("No valid data to group by location for delay.")
        return

    # Sort descending by delay
    sorted_delays = group.sort_values(ascending=False).head(top_n)

    plt.figure()
    plt.bar(sorted_delays.index.astype(str), sorted_delays.values)
    plt.xlabel("Location")
    plt.ylabel("Total Delay (minutes)")
    plt.title(f"Top {top_n} Locations with Most Total Delay")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_day_of_week_most_delay(df, day_col="day_of_week", delay_col="delay_minutes"):
    """
    Bar chart showing the total delay by day of week (over one year or entire dataset).
    Insight:
      - Highlights which weekday or weekend day accumulates the highest delays.
    Note:
      - Ensure 'day_of_week' is a string or category (e.g., Monday, Tuesday, etc.)
        or convert day numbers to names before plotting.
    """
    group = df.dropna(subset=[day_col, delay_col]).groupby(day_col)[delay_col].sum()
    if group.empty:
        print("No valid data to group by day_of_week for delay.")
        return

    # For a nicer x-axis order, map day names to a known sequence if needed:
    # example ordering if your dataset uses full names:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # Filter out any days not present in the data
    day_order = [d for d in day_order if d in group.index]

    # Reindex to ensure a consistent M->Sun order
    group = group.reindex(day_order)

    plt.figure()
    plt.bar(group.index.astype(str), group.values)
    plt.xlabel("Day of Week")
    plt.ylabel("Total Delay (minutes)")
    plt.title("Total Delay by Day of Week")
    plt.grid(True)
    plt.show()


def plot_relation_weather_delay(
    df,
    weather_col="temperature_2m",
    delay_col="delay_minutes",
    bins=20
):
    """
    Displays a scatter of average delay vs. weather measurement (e.g., temperature).
    Insight:
      - Observe if higher/lower temperatures tend to correlate with larger delays.
    Steps:
      1) Bin the temperature range to reduce noise.
      2) For each bin, compute the mean (or sum) delay.
      3) Scatter/bars of binned temperature vs. average delay.
    """
    # Drop rows with missing data
    data = df.dropna(subset=[weather_col, delay_col]).copy()
    if data.empty:
        print(f"No valid data to relate {weather_col} and {delay_col}.")
        return

    # Bin the weather column
    data["weather_bin"] = pd.cut(data[weather_col], bins=bins)
    grouped = data.groupby("weather_bin")[delay_col].mean()

    # The x-values for plotting: midpoints of each bin
    midpoints = []
    for interval in grouped.index:
        # interval is something like (0.0, 5.0], we want ~2.5
        mid = (interval.left + interval.right) / 2.0
        midpoints.append(mid)

    # Build the scatter
    plt.figure()
    plt.scatter(midpoints, grouped.values, alpha=0.7)
    plt.xlabel(f"Binned {weather_col}")
    plt.ylabel(f"Avg {delay_col}")
    plt.title(f"Average Delay vs. {weather_col} (binned)")
    plt.grid(True)
    plt.show()


def plot_when_most_delay_per_day(df, time_col="timestamp", delay_col="delay_minutes"):
    """
    Plots a bar chart of total delay by hour-of-day.
    Insight:
      - Identifies peak hours for delays within a single day or across the entire dataset.
    Note:
      - For a single year's data, grouping by hour is typically enough to see daily patterns.
    """
    # Create an hour column
    df["hour"] = df[time_col].dt.hour
    group = df.dropna(subset=["hour", delay_col]).groupby("hour")[delay_col].sum()
    if group.empty:
        print("No valid data to group by hour of day.")
        return

    plt.figure()
    plt.bar(group.index.astype(str), group.values)
    plt.xlabel("Hour of Day (0-23)")
    plt.ylabel("Total Delay (minutes)")
    plt.title("When Most Delay Occurred (By Hour of Day)")
    plt.grid(True)
    plt.show()

def make_risk_map_for_all_modes(
    df,
    clf_model,
    feature_cols=None,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="all_modes_risk_map.html"
):
    """
    Generates a Folium-based risk map for all transit modes in the dataset.
    Steps:
      1) Uses 'predict_risk_for_map' to compute probability of delay for each row.
      2) Calls 'plot_risk_map' to display the results on a map with color-coded markers.
    Inputs:
      df           : DataFrame containing your enriched data.
      clf_model    : A trained classifier that implements 'predict_proba'.
      feature_cols : List of features that your model expects (e.g. ['hour', 'month', 'is_weekend', 'mode', 'route_or_line']).
      lat_col      : Name of the latitude column in df (defaults to 'stop_lat').
      lon_col      : Name of the longitude column in df (defaults to 'stop_lon').
      location_col : Name of a descriptive location column in df (defaults to 'location').
      output_html  : HTML file to save the interactive map.
    """
    # 1) Compute predicted risk
    risk_df = predict_risk_for_map(
        df=df,
        clf_model=clf_model,
        feature_cols=feature_cols,
        lat_col=lat_col,
        lon_col=lon_col
    )
    if risk_df.empty:
        print("No risk data returned. Check if your DataFrame has the required columns.")
        return None

    # 2) Plot the map
    m = plot_risk_map(
        risk_df,
        location_col=location_col,
        output_html=output_html
    )
    return m