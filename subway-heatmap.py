import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the data
df = pd.read_csv('subway-data-with-coordinates.csv')

# Clean the data
df = df.dropna(subset=['Latitude', 'Longitude'])

# Create a pivot table to count incidents by station
station_counts = df.groupby(['Station']).agg({
    'Latitude': 'first',
    'Longitude': 'first',
    'Station': 'size'
}).rename(columns={'Station': 'Incidents'})

# Create the base map centered on Toronto
m = folium.Map(
    location=[43.7, -79.4],
    zoom_start=11,
    tiles='cartodbpositron'
)

# Create the heatmap data
heat_data = [[row['Latitude'], row['Longitude'], row['Incidents']]
             for idx, row in station_counts.iterrows()]

# Add the heatmap layer
HeatMap(
    heat_data,
    radius=15,
    min_opacity=0.5,
    blur=10,
    max_zoom=1
).add_to(m)

# Add markers for each station
for idx, row in station_counts.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        popup=f"{idx}<br>Incidents: {row['Incidents']}",
        color='black',
        fill=True,
        fill_color='red' if row['Incidents'] > station_counts['Incidents'].quantile(0.75) else 'blue',
        fill_opacity=0.7,
        opacity=0.8
    ).add_to(m)

# Save the map
m.save('ttc_heatmap.html')

print("Heatmap has been created and saved as 'ttc_heatmap.html'")