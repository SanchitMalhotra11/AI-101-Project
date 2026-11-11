"""Reusable helpers for the Fake Bills Detection project.

Everything here is plain pandas / NumPy / scikit-learn / matplotlib.
Models are saved with the standard-library ``pickle`` module (no joblib).
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # safe for tests / headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TARGET_COLUMN = "is_genuine"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(path, target=TARGET_COLUMN):
    """Load the fake-bills CSV and return a DataFrame with a 0/1 target column.

    The delimiter (``;`` or ``,``) is detected automatically.
    Genuine bills -> 1, fake bills -> 0.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    col = df[target]
    if not pd.api.types.is_numeric_dtype(col):
        col = col.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        )
    df[target] = col.astype(int)
    return df


def split_features_target(df, target=TARGET_COLUMN):
    """Return (X, y) from a DataFrame."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(y_true, y_pred, y_score=None):
    """Return a dict of classification metrics.

    Keys: Accuracy, Precision, Recall, F1 (and ROC_AUC when ``y_score`` is given).
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_true, y_score)
    return metrics


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None, labels=("Fake", "Genuine")):
    """Plot (and optionally save) a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    return _finish(fig, save_path)


def plot_roc_curve(y_true, y_score, save_path=None):
    """Plot (and optionally save) the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _finish(fig, save_path)


def plot_feature_importance(feature_names, importances, save_path=None, title="Feature Importance"):
    """Horizontal bar chart of feature importances / coefficients."""
    importances = np.asarray(importances)
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_names)[order], importances[order], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _finish(fig, save_path)


def plot_model_comparison(results, save_path=None):
    """Bar chart comparing models. ``results`` = {model_name: mean_cv_score}."""
    names = list(results.keys())
    scores = list(results.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, scores, color="steelblue")
    ax.set_ylim(max(0.0, min(scores) - 0.05), 1.0)
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title("Model Comparison")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _finish(fig, save_path)


# --------------------------------------------------------------------------- #
# Saving / loading models (pickle, not joblib)
# --------------------------------------------------------------------------- #
def save_model(model, path):
    """Save a fitted model/pipeline with pickle."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Load a model saved with :func:`save_model`."""
    with open(path, "rb") as f:
        return pickle.load(f)
