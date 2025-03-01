########################################
# COMBINED PIPELINE: DATA PREPARATION & MODELING
########################################

########################################
# STEP 1: IMPORT LIBRARIES
########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings

########################################
# STEP 2: LOAD RAW DATA
########################################
# Replace these file paths with the paths on your machine.
bus_df = pd.read_csv('bus-data.csv/Users/christinteng/PycharmProjects/datathon/bus-data.csv')
streetcar_df = pd.read_csv('/Users/christinteng/PycharmProjects/datathon/streetcar-data.csv')
subway_df = pd.read_csv('/Users/christinteng/PycharmProjects/datathon/subway-data.csv')
subway_codes_df = pd.read_csv('/Users/christinteng/PycharmProjects/datathon/subway-delay-codes.csv')

# External data placeholders (update with your actual data)
weather_df = pd.read_csv('weather_data.csv')  # e.g., Date, Temp, Rain, Snow
events_df = pd.read_csv('events_data.csv')  # e.g., Date, EventName, EventType
ridership_df = pd.read_csv('ridership_data.csv')  # e.g., Date, RidershipCount

print("Raw Bus data shape:", bus_df.shape)
print("Raw Streetcar data shape:", streetcar_df.shape)
print("Raw Subway data shape:", subway_df.shape)


########################################
# STEP 3: DATA CLEANING & STANDARDIZATION
########################################

# --- Function for Bus and Streetcar Data ---
def clean_bus_streetcar(df, mode='Bus'):
    """
    Cleans bus or streetcar data:
      - Converts 'Date' to datetime.
      - Combines 'Date' and 'Time' into 'DateTime' (assuming time is in "hh:mm:ss AM/PM").
      - Strips extra spaces from text fields.
      - Converts numeric columns (Min Delay, Min Gap) to numbers.
      - Adds a 'Mode' column.
    """
    df = df.copy()

    # Convert Date field
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')
    # Clean Time field (strip extra spaces)
    df['Time'] = df['Time'].str.strip()
    # Combine Date and Time into DateTime column
    df['DateTime'] = df.apply(
        lambda row: pd.to_datetime(str(row['Date'].date()) + ' ' + row['Time'],
                                   format='%Y-%m-%d %I:%M:%S %p', errors='coerce'),
        axis=1
    )
    # Clean text fields
    df['Location'] = df['Location'].str.strip()
    df['Incident'] = df['Incident'].str.strip()
    # Convert numeric fields
    df['Min Delay'] = pd.to_numeric(df['Min Delay'], errors='coerce')
    df['Min Gap'] = pd.to_numeric(df['Min Gap'], errors='coerce')
    # Add mode column
    df['Mode'] = mode
    # Drop rows missing essential info
    df.dropna(subset=['DateTime', 'Min Delay'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


bus_clean = clean_bus_streetcar(bus_df, mode='Bus')
streetcar_clean = clean_bus_streetcar(streetcar_df, mode='Streetcar')


# --- Function for Subway Data ---
def clean_subway(df, codes_df):
    """
    Cleans subway data:
      - Converts 'Date' to datetime.
      - Parses 'Time' (assumed to be in 24h format) and combines with Date into 'DateTime'.
      - Merges with delay codes (subway_codes_df) to translate 'Code' into a descriptive 'Incident'.
      - Converts numeric fields.
      - Adds a 'Mode' column.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')