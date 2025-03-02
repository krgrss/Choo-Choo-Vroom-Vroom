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

from visualization import (
    predict_risk_for_map, plot_risk_map, 
    create_heatmap_with_markers, create_subway_heatmap_from_excel,
    plot_actual_vs_predicted,
    plot_most_delay_by_mode,
    plot_place_most_delay,
    plot_day_of_week_most_delay,
    plot_relation_weather_delay,
    plot_when_most_delay_per_day,
    make_risk_map_for_all_modes
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

    # A) RISK MAP using your XGBoost classifier's predicted probability
    # ------------------------------------------------------------
    #  If you want to see "predicted risk" of delay at each lat/lon:
    risk_df = predict_risk_for_map(enriched_df, clf_model)
    # risk_df will have columns: [lat, lon, predicted_risk, mode, location, ...]
    # Now we plot it:
    plot_risk_map(risk_df, location_col="location", output_html="bus_risk_map.html")
    # => This saves an HTML map at bus_risk_map.html

    # B) HEATMAP for all modes (bus/streetcar/subway) from enriched_df
    # ------------------------------------------------------------
    # We'll do a simple incident-based heatmap. "create_heatmap_with_markers" groups by "location"
    # and sums how many rows appear at each lat/lon.
    create_heatmap_with_markers(
        df=enriched_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        location_col="location",
        output_html="all_modes_heatmap.html"
    )
    # => This saves "all_modes_heatmap.html"

    # C) SUBWAY LAT/LON from an Excel file
    # ------------------------------------------------------------
    # If you specifically want to create a separate map using the "subway_latlon.xlsx" approach:
    create_subway_heatmap_from_excel(
        incidents_csv="data/subway-data.csv",
        latlon_xlsx="data/subway_latlon.xlsx",
        output_html="subway_heatmap.html"
    )
    # => This merges your raw subway CSV with station coords from the Excel, 
    #    then produces "subway_heatmap.html".

    make_risk_map_for_all_modes(
        df=enriched_df,
        clf_model=clf_model,
        feature_cols=["hour", "month", "is_weekend", "mode", "route_or_line"],
        output_html="all_modes_risk_map.html"
    )

    create_heatmap_with_markers(
        df=enriched_df,
        lat_col="stop_lat",
        lon_col="stop_lon",
        location_col="location",
        output_html="all_modes_heatmap.html"
    )

    plot_actual_vs_predicted(combined_df, actual_col="delay_minutes", predicted_col="my_model_prediction")
    plot_most_delay_by_mode(combined_df)
    plot_place_most_delay(combined_df)
    plot_day_of_week_most_delay(combined_df)
    plot_relation_weather_delay(combined_df, weather_col="temperature_2m")
    plot_when_most_delay_per_day(combined_df, time_col="timestamp")

    print("=== Pipeline and visualization steps complete. ===")

if __name__ == "__main__":
    main()
