import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix, classification_report
import folium
from folium.plugins import MarkerCluster


def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix for classification tasks.
    'classes' can be a list of class names, e.g. ['No Delay', 'Delay'] 
    if y is binary.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    
    if classes is not None:
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticklabels(classes)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # Show values in cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color=color)
    
    plt.tight_layout()
    plt.show()

def regression_error_plot(y_true, y_pred, title="Regression Predictions vs Actual"):
    """
    Scatter plot comparing predicted vs. actual values for a regression task.
    """
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.3)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(title)
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.show()

def shap_summary_plot(xgb_model, X, feature_names=None):
    """
    Generates a SHAP summary plot for the trained XGBoost model.
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)

def plot_time_series_predictions(timestamps, actual, predicted, title="Time Series: Predictions vs Actual"):
    """
    Plots a time-series line chart for actual vs predicted values.
    'timestamps' is an array-like of datetime values.
    """
    plt.figure(figsize=(12,6))
    plt.plot(timestamps, actual, label="Actual", marker='o')
    plt.plot(timestamps, predicted, label="Predicted", marker='x')
    plt.xlabel("Time")
    plt.ylabel("Delay (minutes)")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_geographical_route_risk(route_data, title="Geographical Route-Based Delay Risk"):
    """
    Displays an interactive map with markers for each route (or stop) colored by predicted delay risk.
    'route_data' is expected to be a list of dictionaries with keys:
      'location' : name,
      'lat' : latitude,
      'lon' : longitude,
      'risk' : a numeric value representing risk (e.g., predicted frequency of delay).
    """
    # Center map on Toronto
    toronto_center = [43.6532, -79.3832]
    base_map = folium.Map(location=toronto_center, zoom_start=12)
    
    # Optionally use MarkerCluster to cluster many markers
    marker_cluster = MarkerCluster().add_to(base_map)
    
    # Define a simple risk-to-color mapping: higher risk -> red, lower -> green.
    # You might scale your risk values accordingly.
    def get_color(risk, min_risk, max_risk):
        # Normalize risk to [0,1]
        norm = (risk - min_risk) / (max_risk - min_risk + 1e-5)
        # Use green (low) to red (high)
        if norm < 0.33:
            return "green"
        elif norm < 0.66:
            return "orange"
        else:
            return "red"
    
    risks = [d['risk'] for d in route_data]
    min_risk, max_risk = min(risks), max(risks)
    
    for entry in route_data:
        lat, lon = entry['lat'], entry['lon']
        risk = entry['risk']
        popup_text = f"Location: {entry['location']}<br>Risk: {risk}"
        color = get_color(risk, min_risk, max_risk)
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=color)
        ).add_to(marker_cluster)
    
    base_map.save("geographical_route_risk.html")
    print("Geographical route risk map saved as 'geographical_route_risk.html'.")
    return base_map
