import pandas as pd

from data_preprocessing import main_preprocessing
from feature_engineering import feature_engineering_pipeline
from model_training import (
    prepare_classification_data, train_classification_model,
    prepare_regression_data, train_regression_model
)
from evaluation import plot_confusion_matrix, regression_error_plot
from sklearn.metrics import confusion_matrix

def main():
    # 1) Load and unify the bus, streetcar, and subway data
    combined_df = main_preprocessing(
        bus_path="data/bus-data.csv",
        streetcar_path="data/streetcar-data.csv",
        subway_path="data/subway-data.csv",
        codes_path="data/subway-delay-codes.csv"
    )
    print(f"Data loaded. Shape: {combined_df.shape}")

    # 2) Feature engineering (time-based, weather, events, etc.)
    combined_df = feature_engineering_pipeline(combined_df)
    print(f"After feature engineering. Columns: {combined_df.columns.tolist()}")

    # 3) Classification approach (Delay vs No Delay)
    X_clf, y_clf, df_clf = prepare_classification_data(combined_df)
    print(f"Classification dataset size: X={X_clf.shape}, y={y_clf.shape}")
    classifier = train_classification_model(X_clf, y_clf)

    # Optional: Evaluate classification with a confusion matrix plot
    y_clf_pred = classifier.predict(X_clf)
    cm = confusion_matrix(y_clf, y_clf_pred)
    print("Full confusion matrix on entire dataset:\n", cm)
    # If you want a nice plot:
    # plot_confusion_matrix(y_clf, y_clf_pred, classes=["No Delay", "Delay"])

    # 4) Regression approach (Predict how many minutes the delay will be)
    X_reg, y_reg, df_reg = prepare_regression_data(combined_df)
    print(f"Regression dataset size: X={X_reg.shape}, y={y_reg.shape}")
    regressor = train_regression_model(X_reg, y_reg)

    # Optional: scatter plot of predictions vs actual
    y_reg_pred = regressor.predict(X_reg)
    # regression_error_plot(y_reg, y_reg_pred, title="Delay Duration: Predicted vs Actual")

    print("=== Pipeline complete ===")

if __name__ == "__main__":
    main()
