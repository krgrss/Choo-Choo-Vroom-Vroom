import pandas as pd
import numpy as np

# Data Preprocessing
from data_preprocessing import main_preprocessing

# Feature Engineering
from feature_engineering import feature_engineering_pipeline

# Model Training
from model_training import (
    prepare_classification_data, train_classification_model,
    prepare_regression_data, train_regression_model
)

# Optional evaluation / visualization
from evaluation import (
    plot_confusion_matrix, regression_error_plot,
    plot_time_series_predictions, plot_geographical_route_risk
)
from sklearn.metrics import confusion_matrix

def main():
    """
    1) Data Preprocessing -> combine bus, streetcar, subway, references stops.txt
    2) Feature Engineering -> add weather/time/lag/fuzzy route if needed
    3) Prepare & train classification/regression models
    """
    print("=== 1) Data Preprocessing ===")
    combined_df = main_preprocessing(
        bus_path="data/bus-data.csv",
        streetcar_path="data/streetcar-data.csv",
        subway_path="data/subway-data.csv",
        codes_path="data/subway-delay-codes.csv",
        stops_path="data/external/gtfs/stops.txt"
    )
    print(f"Combined data shape: {combined_df.shape}")

    print("=== 2) Feature Engineering ===")
    enriched_df = feature_engineering_pipeline(combined_df)
    print("Feature engineering complete. Columns now include:")
    print(enriched_df.columns.tolist())

    # Save to CSV
    enriched_df.to_csv("enriched_data.csv", index=False)
    print("Enriched data saved to 'enriched_data.csv'. Shape:", enriched_df.shape)

    print("\n=== 3) Model Training: Classification ===")
    X_clf, y_clf, df_clf = prepare_classification_data(enriched_df)
    print(f"Classification dataset size: X={X_clf.shape}, y={y_clf.shape}")
    if X_clf.shape[0] == 0:
        print("No samples for classification.")
        return
    clf_model = train_classification_model(X_clf, y_clf)
    y_clf_pred = clf_model.predict(X_clf)
    cm = confusion_matrix(y_clf, y_clf_pred)
    print("Confusion matrix on classification dataset:\n", cm)
    # plot_confusion_matrix(y_clf, y_clf_pred, classes=["No Delay", "Delay"])

    print("\n=== 3) Model Training: Regression ===")
    X_reg, y_reg, df_reg = prepare_regression_data(enriched_df)
    print(f"Regression dataset size: X={X_reg.shape}, y={y_reg.shape}")
    if X_reg.shape[0] == 0:
        print("No samples for regression.")
        return
    reg_model = train_regression_model(X_reg, y_reg)

    # Optionally, add example time-series / route risk visualization
    ts_data = {
        "timestamp": pd.date_range("2024-01-01", periods=24, freq="H"),
        "actual_delay": [10, 5, 0, 12, 8, 15, 5, 3, 20, 10, 7, 5, 8, 12, 15, 0, 4, 10, 6, 8, 12, 5, 7, 10],
        "predicted_delay": [8, 7, 0, 10, 9, 13, 6, 4, 18, 11, 8, 6, 10, 11, 14, 0, 3, 9, 7, 10, 11, 4, 6, 9]
    }
    ts_df = pd.DataFrame(ts_data)
    plot_time_series_predictions(ts_df["timestamp"], ts_df["actual_delay"], ts_df["predicted_delay"])

    route_data = [
        {"location": "Route 1", "lat": 43.67, "lon": -79.39, "risk": 0.8},
        {"location": "Route 2", "lat": 43.65, "lon": -79.38, "risk": 0.3},
        {"location": "Route 3", "lat": 43.66, "lon": -79.40, "risk": 0.6},
    ]
    map_obj = plot_geographical_route_risk(route_data)
    map_obj.save("geographical_route_risk.html")
    print("Geographical route risk map saved to 'geographical_route_risk.html'.")

    print("\n=== Pipeline complete ===")

if __name__ == "__main__":
    main()
