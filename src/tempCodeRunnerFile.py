import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error
from data_preprocessing import main_preprocessing

from feature_engineering import feature_engineering_pipeline

# Model training module with classification/regression data prep and XGBoost training
from model_training import (
    prepare_classification_data, train_classification_model,
    prepare_regression_data, train_regression_model
)

from evaluation import (
    plot_confusion_matrix, regression_error_plot,
    plot_time_series_predictions, plot_geographical_route_risk
)
def prepare_classification_data(df, delay_threshold=1):
    """
    Prepares data for a binary classification task.
    A record is marked as a delay event (1) if delay_minutes >= delay_threshold, else 0.
    
    Features used include:
      - hour, month, is_weekend, mode, and a unified 'route_or_line'
    
    If the 'route' column is missing or has only empty values, 'location' is used instead.
    """
    # Create the binary target variable.
    df["delay_binary"] = (df["delay_minutes"] >= delay_threshold).astype(int)
    
    # Decide which column to use as route source.
    # Check if "route" exists and if at least one value is non-empty after stripping.
    if "route" in df.columns:
        route_series = df["route"].astype(str).str.strip().replace("", pd.NA)
        if route_series.notna().sum() > 0:
            route_data = route_series
        else:
            route_data = df["location"].astype(str).str.strip()
    else:
        route_data = df["location"].astype(str).str.strip()
    
    # If 'line' exists, fill missing values from the chosen column.
    if "line" in df.columns:
        line_series = df["line"].astype(str).str.strip().replace("", pd.NA)
        df["route_or_line"] = route_data.fillna(line_series)
    else:
        df["route_or_line"] = route_data
    
    # Drop rows where route_or_line is missing or empty.
    df = df[df["route_or_line"].astype(str).str.strip() != ""]
    
    # Define feature columns.
    feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]
    
    # Prepare X and y.
    y = df["delay_binary"].values
    X_data = df[feature_cols].copy()
    
    # Label encode categorical features.
    from sklearn.preprocessing import LabelEncoder
    for col in ["mode", "route_or_line"]:
        le = LabelEncoder()
        X_data[col] = le.fit_transform(X_data[col].astype(str))
    
    X = X_data.values
    return X, y, df

def prepare_regression_data(df):
    """
    Prepares data for regression to predict delay_minutes.
    Only includes rows with delay_minutes > 0.
    
    Uses features:
      - hour, month, is_weekend, mode, and a unified 'route_or_line'
    
    If 'route' is missing or empty, 'location' is used instead.
    """
    df_reg = df[df["delay_minutes"] > 0].copy()
    
    if "route" in df_reg.columns:
        route_series = df_reg["route"].astype(str).str.strip().replace("", pd.NA)
        if route_series.notna().sum() > 0:
            route_data = route_series
        else:
            route_data = df_reg["location"].astype(str).str.strip()
    else:
        route_data = df_reg["location"].astype(str).str.strip()
    
    if "line" in df_reg.columns:
        line_series = df_reg["line"].astype(str).str.strip().replace("", pd.NA)
        df_reg["route_or_line"] = route_data.fillna(line_series)
    else:
        df_reg["route_or_line"] = route_data
    
    df_reg = df_reg[df_reg["route_or_line"].astype(str).str.strip() != ""]
    
    feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]
    
    from sklearn.preprocessing import LabelEncoder
    for col in ["mode", "route_or_line"]:
        le = LabelEncoder()
        df_reg[col] = le.fit_transform(df_reg[col].astype(str))
    
    X = df_reg[feature_cols].values
    y = df_reg["delay_minutes"].values
    return X, y, df_reg


def train_classification_model(X, y):
    """
    Splits the data, trains an XGBoost classifier, and prints evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))
    
    return clf

def train_regression_model(X, y):
    """
    Splits the data, trains an XGBoost regressor, and prints the Mean Absolute Error.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    reg = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    reg.fit(X_train, y_train)
    
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("=== XGBoost Regression MAE ===")
    print(f"{mae:.2f} minutes")
    
    return reg

if __name__ == "__main__":
    combined_df = main_preprocessing(
        bus_path="data/bus-data.csv",
        streetcar_path="data/streetcar-data.csv",
        subway_path="data/subway-data.csv",
        codes_path="data/subway-delay-codes.csv"
    )
    # 2) Feature Engineering
    enriched_df = feature_engineering_pipeline(combined_df)
    enriched_df.to_csv("enriched_data.csv", index=False)
    
    # Prepare and train the classification model.
    print("Preparing classification data...")
    X_clf, y_clf, df_clf = prepare_classification_data(enriched_df)
    print("Training classification model...")
    clf_model = train_classification_model(X_clf, y_clf)
    
    # Prepare and train the regression model.
    print("Preparing regression data...")
    X_reg, y_reg, df_reg = prepare_regression_data(enriched_df)
    print("Training regression model...")
    reg_model = train_regression_model(X_reg, y_reg)
