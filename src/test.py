import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_absolute_error

def prepare_classification_data(df):
    """
    Example approach:
    We'll define a classification target: 
    'Will there be a delay >= 1 minute on this route in the next hour?'
    --------------------------------------------------------------------
    But you can adapt as needed. This is just to show how you might 
    set up the classification problem.
    """

    # 1) For demonstration, let's consider each record an event. 
    #    We'll say y=1 if delay_minutes >= 1, else 0.
    #    (In practice, you might do something more time-series oriented.)
    df["delay_binary"] = (df["delay_minutes"] >= 1).astype(int)
    
    # 2) Features: we pick some columns (hour, is_weekend, route/line, etc.)
    feature_cols = [
        "hour", 
        "month",
        "is_weekend",
        "mode"
    ]
    # We'll also consider route or line as categorical, but we can pick one. 
    # For instance, let's use route if mode is "bus" or "streetcar"; 
    # for subway, route is NaN, so we might use line. 
    # A simple approach: fill route with line when route is NaN:
    df["route_or_line"] = df["route"].fillna(df["line"])
    feature_cols.append("route_or_line")

    # Drop rows where route_or_line is still NaN (should be rare if data is valid)
    df = df.dropna(subset=["route_or_line"])

    # 3) Prepare X, y
    y = df["delay_binary"].values

    # Convert the feature set to numeric, label-encoding categorical fields
    X_data = df[feature_cols].copy()

    # We'll transform 'mode' and 'route_or_line' using label encoding
    for col in ["mode", "route_or_line"]:
        le = LabelEncoder()
        X_data[col] = le.fit_transform(X_data[col].astype(str))

    X = X_data.values

    return X, y, df


def train_classification_model(X, y):
    """
    Train an XGBoost classifier to predict delay occurrence (y=1 or 0).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Basic XGBoost classifier. Tune hyperparameters as needed.
    xgb_clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_clf.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb_clf.predict(X_test)
    print("=== XGBoost Classification Report ===")
    print(classification_report(y_test, y_pred))

    return xgb_clf


def prepare_regression_data(df):
    """
    Prepare data for a regression task: predict 'delay_minutes' if there's any delay.
    We'll exclude zero-delay events (if they exist) since there's nothing to predict 
    (or you can keep them if you want to model them).
    """

    # Filter to rows that actually have a delay > 0
    df_reg = df[df["delay_minutes"] > 0].copy()

    # Example features for regression:
    feature_cols = [
        "hour", 
        "month",
        "is_weekend",
        "mode"
    ]

    # Similar logic: let's unify route or line
    df_reg["route_or_line"] = df_reg["route"].fillna(df_reg["line"])
    df_reg = df_reg.dropna(subset=["route_or_line"])
    feature_cols.append("route_or_line")

    # Label encode the categorical fields
    for col in ["mode", "route_or_line"]:
        le = LabelEncoder()
        df_reg[col] = le.fit_transform(df_reg[col].astype(str))

    X = df_reg[feature_cols].values
    y = df_reg["delay_minutes"].values

    return X, y, df_reg


def train_regression_model(X, y):
    """
    Train an XGBoost regressor to predict delay duration (minutes).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2, 
                                                        random_state=42)

    xgb_reg = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_reg.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = xgb_reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"=== XGBoost Regression MAE ===\n{mae:.2f} minutes")

    return xgb_reg
