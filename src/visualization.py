import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt

##############################################################################
# PART A: MATPLOTLIB VISUALIZATIONS
##############################################################################

def plot_actual_vs_predicted(
    df,
    actual_col="delay_minutes",
    predicted_col="predicted_delay"
):
    """
    Scatter plot of Actual vs. Predicted delay.
    Insight:
      - Points along the diagonal indicate close agreement.
      - Points far from the diagonal show where prediction was off.
    """
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
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual Delay")
    plt.ylabel("Predicted Delay")
    plt.title("Actual vs. Predicted Delay")
    plt.grid(True)
    plt.show()


def plot_most_delay_by_mode(
    df,
    mode_col="mode",
    delay_col="delay_minutes"
):
    """
    Bar chart of total delay by each transit mode.
    Insight:
      - Identifies which mode accumulates the most total delay.
    """
    grouped = df.dropna(subset=[mode_col, delay_col]).groupby(mode_col)[delay_col].sum()
    if grouped.empty:
        print("No valid data to group by mode for delay.")
        return

    grouped = grouped.sort_values(ascending=False)
    plt.figure()
    plt.bar(grouped.index.astype(str), grouped.values)
    plt.xlabel("Mode")
    plt.ylabel("Total Delay (minutes)")
    plt.title("Most Delay by Mode")
    plt.grid(True)
    plt.show()


def plot_place_most_delay(
    df,
    location_col="location",
    delay_col="delay_minutes",
    top_n=10
):
    """
    Bar chart showing which places have the highest total delay.
    Insight:
      - Useful for spotting worst-performing stops/stations.
    """
    group = df.dropna(subset=[location_col, delay_col]).groupby(location_col)[delay_col].sum()
    if group.empty:
        print("No valid data to group by location for delay.")
        return

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


def plot_day_of_week_most_delay(
    df,
    day_col="day_of_week",
    delay_col="delay_minutes"
):
    """
    Bar chart showing total delay by day of week.
    Insight:
      - Reveals which weekday/weekend day has highest delays.
    """
    group = df.dropna(subset=[day_col, delay_col]).groupby(day_col)[delay_col].sum()
    if group.empty:
        print("No valid data to group by day_of_week for delay.")
        return

    # Example ordering if you have full day names:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_order = [d for d in day_order if d in group.index]
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
    Displays a scatter of average delay vs. binned weather measurement (e.g. temperature).
    Insight:
      - Observe if extremes in weather correlate with larger delays.
    """
    data = df.dropna(subset=[weather_col, delay_col]).copy()
    if data.empty:
        print(f"No valid data to relate {weather_col} and {delay_col}.")
        return

    data["weather_bin"] = pd.cut(data[weather_col], bins=bins)
    grouped = data.groupby("weather_bin")[delay_col].mean()

    # Midpoints of each bin
    midpoints = []
    for interval in grouped.index:
        mid = (interval.left + interval.right) / 2.0
        midpoints.append(mid)

    plt.figure()
    plt.scatter(midpoints, grouped.values, alpha=0.7)
    plt.xlabel(f"Binned {weather_col}")
    plt.ylabel(f"Avg {delay_col}")
    plt.title(f"Average Delay vs. {weather_col} (binned)")
    plt.grid(True)
    plt.show()


def plot_when_most_delay_per_day(
    df,
    time_col="timestamp",
    delay_col="delay_minutes"
):
    """
    Bar chart of total delay by hour of day (0-23).
    Insight:
      - Identifies peak hours for delays.
    """
    if time_col not in df.columns:
        print(f"Column '{time_col}' not found in DataFrame.")
        return

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


##############################################################################
# PART B: FOLIUM MAP VISUALIZATIONS (Risk Maps, Heatmaps)
##############################################################################

def predict_risk_for_map(df, clf_model, feature_cols=None, lat_col="stop_lat", lon_col="stop_lon"):
    """
    Takes an enriched DataFrame (df) that has:
      - hour, month, is_weekend, mode, route_or_line (by default)
      - lat/lon columns (default: 'stop_lat', 'stop_lon')
    and a trained classifier (clf_model) with predict_proba.
    
    Returns df_map with columns: [lat, lon, predicted_risk, location, etc.].
    """
    from sklearn.preprocessing import LabelEncoder

    if feature_cols is None:
        feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"No lat/lon columns found: {lat_col}, {lon_col}.")
        return pd.DataFrame()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("Risk map: missing feature columns:", missing)
        return pd.DataFrame()

    df_map = df.copy()

    le_mode = LabelEncoder()
    le_route = LabelEncoder()
    df_map["mode_enc"] = le_mode.fit_transform(df_map["mode"].astype(str))
    df_map["route_or_line_enc"] = le_route.fit_transform(df_map["route_or_line"].astype(str))

    X_map = pd.DataFrame({
        "hour": df_map["hour"],
        "month": df_map["month"],
        "is_weekend": df_map["is_weekend"],
        "mode_enc": df_map["mode_enc"],
        "route_or_line_enc": df_map["route_or_line_enc"]
    })

    if hasattr(clf_model, "predict_proba"):
        df_map["predicted_risk"] = clf_model.predict_proba(X_map)[:, 1]
    else:
        print("Classifier does not have predict_proba.")
        return pd.DataFrame()

    df_map["lat"] = df_map[lat_col]
    df_map["lon"] = df_map[lon_col]
    df_map.dropna(subset=["lat", "lon", "predicted_risk"], inplace=True)
    return df_map


def plot_risk_map(risk_df, location_col="location", output_html="risk_map.html"):
    """
    Creates a Folium map from 'risk_df' (which is the output of 'predict_risk_for_map').
    Markers are color-coded by risk.
    """
    if risk_df.empty:
        print("Risk DF is empty, cannot plot.")
        return None

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


def make_bus_risk_map(
    df,
    clf_model,
    feature_cols=None,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="bus_risk_map.html",
    sample_fraction=0.1,  # 10% by default
    random_state=42
):
    """
    Creates a Folium risk map just for Bus mode, 
    sampling the bus rows (sample_fraction) to reduce clutter.
    """
    df_bus = df[df["mode"] == "bus"].copy()
    if df_bus.empty:
        print("No rows for mode='bus'. Cannot generate bus risk map.")
        return None

    # Sample 10% by default
    df_bus = df_bus.sample(frac=sample_fraction, random_state=random_state)

    risk_df = predict_risk_for_map(
        df=df_bus,
        clf_model=clf_model,
        feature_cols=feature_cols,
        lat_col=lat_col,
        lon_col=lon_col
    )
    if risk_df.empty:
        print("Bus risk DataFrame is empty after sampling or missing columns.")
        return None

    m = plot_risk_map(
        risk_df,
        location_col=location_col,
        output_html=output_html
    )
    return m


def make_risk_map_for_all_modes(
    df,
    clf_model,
    feature_cols=None,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="all_modes_risk_map.html",
    sample_fraction=0.1,  # 10% by default
    random_state=42
):
    """
    Generates a Folium-based risk map for all modes, sampling the entire dataset
    by 'sample_fraction' to avoid overcrowding.
    """
    df_sample = df.sample(frac=sample_fraction, random_state=random_state)
    if df_sample.empty:
        print("Sampling returned no data. Adjust 'sample_fraction' or check your dataset.")
        return None

    risk_df = predict_risk_for_map(
        df=df_sample,
        clf_model=clf_model,
        feature_cols=feature_cols,
        lat_col=lat_col,
        lon_col=lon_col
    )
    if risk_df.empty:
        print("Risk DF is empty or missing required columns.")
        return None

    m = plot_risk_map(
        risk_df,
        location_col=location_col,
        output_html=output_html
    )
    return m


def create_heatmap_with_markers(
    df,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="heatmap.html"
):
    """
    Aggregates incidents by location, producing:
      1) A HeatMap weighted by 'incidents'
      2) CircleMarkers for each location
    """
    df = df.dropna(subset=[lat_col, lon_col])
    if df.empty:
        print("No data after dropping missing lat/lon.")
        return None

    grouped = df.groupby(location_col).agg({
        lat_col: "first",
        lon_col: "first",
        location_col: "size"
    }).rename(columns={location_col: "incidents"}).reset_index()

    m = folium.Map(location=[43.7, -79.4], zoom_start=11)
    heat_data = []
    for _, row in grouped.iterrows():
        heat_data.append([row[lat_col], row[lon_col], row["incidents"]])

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


def create_subway_heatmap_from_excel(
    incidents_csv="data/subway-data.csv",
    latlon_xlsx="data/subway_latlon.xlsx",
    output_html="subway_heatmap.html"
):
    """
    Similar to create_heatmap_with_markers, but merges station lat/lon from Excel first.
    Then builds a heatmap & markers specifically for subways.
    """
    df_incidents = pd.read_csv(incidents_csv, dtype=str)
    if df_incidents.empty:
        print(f"No data in {incidents_csv}")
        return None

    df_incidents.rename(columns={"Station": "station"}, inplace=True, errors="ignore")
    df_incidents["station"] = df_incidents["station"].astype(str).str.strip()

    df_latlon = pd.read_excel(latlon_xlsx)
    df_latlon.rename(columns={"Station": "station", "Latitude": "lat", "Longitude": "lon"}, inplace=True)
    df_latlon["lat"] = pd.to_numeric(df_latlon["lat"], errors="coerce")
    df_latlon["lon"] = pd.to_numeric(df_latlon["lon"], errors="coerce")
    df_latlon["station"] = df_latlon["station"].astype(str).str.strip()

    df_merged = pd.merge(
        df_incidents,
        df_latlon[["station", "lat", "lon"]],
        on="station",
        how="left"
    )
    df_merged.dropna(subset=["lat", "lon"], inplace=True)
    if df_merged.empty:
        print("No station-lat/lon matches found.")
        return None

    grouped = df_merged.groupby("station").agg({
        "lat": "first",
        "lon": "first",
        "station": "size"
    }).rename(columns={"station": "incidents"}).reset_index()

    m = folium.Map(location=[43.7, -79.4], zoom_start=11)
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

if __name__ == "__main__":
    print("visualization.py loaded. Refer to function docstrings for usage.")
