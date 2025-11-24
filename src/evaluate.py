from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(name: str, model, X_test, y_test, save_fig: bool = False, fig_path: str | None = None) -> Dict[str, Any]:
    """Compute standard metrics for classification and return as a dict."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fall back to decision function if available
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # convert scores to 0-1 via min-max for AUC fallback
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            y_proba = scores
        else:
            y_proba = None

    metrics: Dict[str, Any] = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    # add confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

    # Optionally save confusion matrix plot
    if save_fig and fig_path is not None:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    return metrics


def save_metrics(metrics_list, path: str) -> None:
    """Save list of metric dicts to a CSV file."""
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)
