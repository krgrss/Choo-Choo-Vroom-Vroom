import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error

def prepare_classification_data(df, delay_threshold=1):
    df["delay_binary"] = (df["delay_minutes"] >= delay_threshold).astype(int)
    
    # pick route if not empty, else location
    if "route" in df.columns:
        route_series = df["route"].astype(str).str.strip().replace("", pd.NA)
        if route_series.notna().sum() > 0:
            route_data = route_series
        else:
            route_data = df["location"].astype(str).str.strip()
    else:
        route_data = df["location"].astype(str).str.strip()

    if "line" in df.columns:
        line_series = df["line"].astype(str).str.strip().replace("", pd.NA)
        df["route_or_line"] = route_data.fillna(line_series)
    else:
        df["route_or_line"] = route_data

    # Fill empty
    df["route_or_line"] = df["route_or_line"].astype(str).str.strip()
    df["route_or_line"].replace("", "unknown", inplace=True)

    feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]
    y = df["delay_binary"].values
    X_data = df[feature_cols].copy()

    le = LabelEncoder()
    for col in ["mode", "route_or_line"]:
        X_data[col] = le.fit_transform(X_data[col].astype(str))

    X = X_data.values
    return X, y, df

def train_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='logloss'
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== XGBoost Classification Report ===")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    return clf

def prepare_regression_data(df):
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

    df_reg["route_or_line"] = df_reg["route_or_line"].astype(str).str.strip()
    df_reg["route_or_line"].replace("", "unknown", inplace=True)

    feature_cols = ["hour", "month", "is_weekend", "mode", "route_or_line"]
    X_data = df_reg[feature_cols].copy()

    le = LabelEncoder()
    for col in ["mode", "route_or_line"]:
        X_data[col] = le.fit_transform(X_data[col].astype(str))

    X = X_data.values
    y = df_reg["delay_minutes"].values
    return X, y, df_reg

def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("=== XGBoost Regression MAE ===")
    print(f"{mae:.2f} minutes")
    return reg

if __name__ == "__main__":
    print("model_training.py: typically used from main.py")
