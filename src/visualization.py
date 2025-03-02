import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt

#####################################
# 1) Risk Map with XGBoost Probability
#####################################

def predict_risk_for_map(df, clf_model, feature_cols=None, lat_col="stop_lat", lon_col="stop_lon"):
    """
    Takes an enriched DataFrame (df) that has:
      - hour, month, is_weekend, mode, route_or_line (by default)
      - lat/lon columns (default: 'stop_lat', 'stop_lon')
    and a trained XGBoost classifier (clf_model) that can do predict_proba().
    
    1) Label-encodes 'mode' and 'route_or_line' to match training.
    2) Builds an X matrix of your classification features.
    3) Gets predicted probability of delay (class=1).
    4) Returns df_map with columns: [lat, lon, predicted_risk, location, etc.].
    """
    from sklearn.preprocessing import LabelEncoder

    if feature_cols is None:
        # Default feature columns (adjust as needed to match your training)
        feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]
    
    # Ensure lat/lon exist
    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"No lat/lon columns found: {lat_col}, {lon_col} in df.")
        return pd.DataFrame()

    # Check for missing required feature columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print("Risk map: missing feature columns:", missing)
        return pd.DataFrame()

    # Make a copy
    df_map = df.copy()

    # Label-encode 'mode' and 'route_or_line'
    le_mode = LabelEncoder()
    le_route = LabelEncoder()

    df_map["mode_enc"] = le_mode.fit_transform(df_map["mode"].astype(str))
    df_map["route_or_line_enc"] = le_route.fit_transform(df_map["route_or_line"].astype(str))

    # Build classification matrix
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

    # Drop rows missing lat/lon or risk
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

    # Center on Toronto (adjust as you like)
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
    
    Steps:
      1) Group by 'location' => count how many rows (incidents).
      2) Use HeatMap for a visual "hot spots" approach.
      3) Add CircleMarkers for clarity.
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
    m = folium.Map(location=[43.7, -79.4], zoom_start=11)

    # Prepare data for HeatMap
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


#####################################
# 3) Using Subway lat/lon from Excel
#####################################

def create_subway_heatmap_from_excel(
    incidents_csv="data/subway-data.csv",
    latlon_xlsx="data/subway_latlon.xlsx",
    output_html="subway_heatmap.html"
):
    """
    Similar to create_heatmap_with_markers, but merges station lat/lon from Excel first.
    Then builds a heatmap & markers specifically for subways.
    """
    # Load incidents
    df_incidents = pd.read_csv(incidents_csv, dtype=str)
    if df_incidents.empty:
        print(f"No data in {incidents_csv}")
        return None

    df_incidents.rename(columns={"Station": "station"}, inplace=True, errors="ignore")
    df_incidents["station"] = df_incidents["station"].astype(str).str.strip()

    # Load station lat/lon from Excel
    df_latlon = pd.read_excel(latlon_xlsx)
    df_latlon.rename(columns={"Station": "station", "Latitude": "lat", "Longitude": "lon"}, inplace=True)
    df_latlon["lat"] = pd.to_numeric(df_latlon["lat"], errors="coerce")
    df_latlon["lon"] = pd.to_numeric(df_latlon["lon"], errors="coerce")
    df_latlon["station"] = df_latlon["station"].astype(str).str.strip()

    # Merge on station name
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

    # Group to get incidents per station
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


###################################
# 4) Modified: Bus Risk Map with Sampling
###################################

def make_bus_risk_map(
    df,
    clf_model,
    feature_cols=None,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="bus_risk_map.html",
    sample_fraction=0.1,  # <-- 10% by default
    random_state=42
):
    """
    Creates a Folium risk map just for Bus mode,
    but now we take a sample of the bus rows (sample_fraction) 
    to avoid plotting too many points at once.
    """
    # Filter only bus rows
    df_bus = df[df["mode"] == "bus"].copy()
    if df_bus.empty:
        print("No rows for mode='bus'. Cannot generate bus risk map.")
        return None

    # SAMPLE: default 10%
    df_bus = df_bus.sample(frac=sample_fraction, random_state=random_state)

    # Compute predicted risk for bus data only
    risk_df = predict_risk_for_map(
        df=df_bus,
        clf_model=clf_model,
        feature_cols=feature_cols,
        lat_col=lat_col,
        lon_col=lon_col
    )
    if risk_df.empty:
        print("Bus risk DataFrame came back empty. Check input columns or model.")
        return None

    # Plot the map
    m = plot_risk_map(
        risk_df,
        location_col=location_col,
        output_html=output_html
    )
    return m


#####################################
# 5) Modified: Risk Map for All Modes with Sampling
#####################################

def make_risk_map_for_all_modes(
    df,
    clf_model,
    feature_cols=None,
    lat_col="stop_lat",
    lon_col="stop_lon",
    location_col="location",
    output_html="all_modes_risk_map.html",
    sample_fraction=0.1,  # <-- 10% by default
    random_state=42
):
    """
    Generates a Folium-based risk map for every mode in df (bus, streetcar, subway, etc.),
    but samples the data to 'sample_fraction' to avoid overcrowding.
    """
    # SAMPLE: default 10% of entire dataset
    df_sample = df.sample(frac=sample_fraction, random_state=random_state)
    if df_sample.empty:
        print("Sampling returned no data. Adjust sample_fraction or check your dataset.")
        return None

    risk_df = predict_risk_for_map(
        df=df_sample,
        clf_model=clf_model,
        feature_cols=feature_cols,
        lat_col=lat_col,
        lon_col=lon_col
    )
    if risk_df.empty:
        print("No risk data returned. Check if your DataFrame has the required columns.")
        return None

    m = plot_risk_map(
        risk_df,
        location_col=location_col,
        output_html=output_html
    )
    return m


#####################################
# Example (for local testing)
#####################################

if __name__ == "__main__":
    # Example usage or local testing (you can remove or adjust as needed).
    print("visualization.py can be imported into your pipeline. Example usage in docstrings.")
