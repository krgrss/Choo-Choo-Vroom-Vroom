import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import plot_importance

from sklearn.metrics import confusion_matrix, classification_report

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
    # 1:1 line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(title)
    plt.xlabel("Actual Delay (minutes)")
    plt.ylabel("Predicted Delay (minutes)")
    plt.show()
