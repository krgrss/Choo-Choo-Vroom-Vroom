import pandas as pd
import matplotlib.pyplot as plt

# Try to load datasets; if files are missing, use dummy data
try:
    bus_data = pd.read_csv('/mnt/data/bus-data.csv')
    streetcar_data = pd.read_csv('/mnt/data/streetcar-data.csv')
    subway_data = pd.read_csv('/mnt/data/subway-data.csv')
    subway_delay_codes = pd.read_csv('/mnt/data/subway-delay-codes.csv')
except Exception as e:
    print("Error reading one or more files:", e)
    # Creating dummy data for demonstration
    bus_data = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=24, freq='H'),
        'delay_duration': [5, 7, 6, 5, 8, 10, 5, 6, 7, 8, 4, 5, 6, 7, 5, 8, 6, 5, 7, 6, 8, 5, 7, 6],
        'route': ['Bus A'] * 24
    })
    
    streetcar_data = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=24, freq='H'),
        'delay_duration': [3, 4, 5, 3, 4, 6, 3, 4, 5, 4, 3, 4, 5, 4, 3, 6, 4, 3, 5, 4, 3, 4, 5, 4],
        'route': ['Streetcar 1'] * 24
    })
    
    subway_data = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=24, freq='H'),
        'delay_duration': [2, 3, 4, 2, 3, 5, 2, 3, 4, 3, 2, 3, 4, 3, 2, 5, 3, 2, 4, 3, 2, 3, 4, 3],
        'station': ['Station X'] * 24,
        'delay_code': [1, 2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3]
    })
    
    subway_delay_codes = pd.DataFrame({
        'delay_code': [1, 2, 3],
        'delay_reason': ['Signal Failure', 'Maintenance', 'Congestion']
    })

# Display sample data
print("Bus Data Sample:")
print(bus_data.head(), "\n")

print("Streetcar Data Sample:")
print(streetcar_data.head(), "\n")

print("Subway Data Sample:")
print(subway_data.head(), "\n")

print("Subway Delay Codes:")
print(subway_delay_codes.head(), "\n")

# -------------------------
# Feature Engineering & EDA
# -------------------------

# Convert timestamp to datetime and extract the hour for bus data
bus_data['timestamp'] = pd.to_datetime(bus_data['timestamp'])
bus_data['hour'] = bus_data['timestamp'].dt.hour

# Example 1: Plot Average Bus Delay Duration by Hour
bus_delay_by_hour = bus_data.groupby('hour')['delay_duration'].mean()
bus_delay_by_hour.plot(kind='bar', title="Average Bus Delay Duration by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delay (min)")
plt.show()

# Convert timestamp to datetime and extract the hour for streetcar data
streetcar_data['timestamp'] = pd.to_datetime(streetcar_data['timestamp'])
streetcar_data['hour'] = streetcar_data['timestamp'].dt.hour

# Example 2: Plot Average Streetcar Delay Duration by Hour
streetcar_delay_by_hour = streetcar_data.groupby('hour')['delay_duration'].mean()
streetcar_delay_by_hour.plot(kind='bar', title="Average Streetcar Delay Duration by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Delay (min)")
plt.show()

# Merge subway data with delay codes to interpret delay reasons
subway_data = subway_data.merge(subway_delay_codes, on='delay_code', how='left')

# Convert timestamp to datetime and extract the hour for subway data
subway_data['timestamp'] = pd.to_datetime(subway_data['timestamp'])
subway_data['hour'] = subway_data['timestamp'].dt.hour

# Example 3: Plot Subway Delay Frequency by Station
station_delay_counts = subway_data['station'].value_counts()
station_delay_counts.plot(kind='bar', title="Subway Delay Frequency by Station")
plt.xlabel("Station")
plt.ylabel("Number of Delays")
plt.show()

# Example 4: Analyze and Plot Frequency of Delay Reasons in Subway Data
delay_reason_counts = subway_data['delay_reason'].value_counts()
delay_reason_counts.plot(kind='bar', title="Frequency of Subway Delay Reasons")
plt.xlabel("Delay Reason")
plt.ylabel("Frequency")
plt.show()
