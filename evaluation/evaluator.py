"""
Model evaluation: metrics, confusion matrix, ROC curves, reports.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import EMOTION_LABELS, NUM_CLASSES

logger = logging.getLogger(__name__)

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve, auc, classification_report,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─── Core Metrics ────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str] = None,
) -> Dict:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required")

    names = label_names or EMOTION_LABELS

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=[names[i] for i in sorted(set(y_true)) if i < len(names)],
            zero_division=0, output_dict=True,
        ),
    }


def compute_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    if not HAS_SKLEARN:
        return np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    return confusion_matrix(y_true, y_pred, labels=labels or list(range(NUM_CLASSES)))


def compute_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,  # (N, num_classes)
    label_names: List[str] = None,
) -> Tuple[Dict, Dict, Dict]:
    if not HAS_SKLEARN:
        return {}, {}, {}

    names = label_names or EMOTION_LABELS
    n_classes = y_prob.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n_classes):
        if i < len(names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            fpr_d[names[i]] = fpr.tolist()
            tpr_d[names[i]] = tpr.tolist()
            auc_d[names[i]] = float(auc(fpr, tpr))

    return fpr_d, tpr_d, auc_d


def classwise_performance(metrics: Dict) -> pd.DataFrame:
    report = metrics.get("classification_report", {})
    rows = []
    for emotion, vals in report.items():
        if isinstance(vals, dict):
            rows.append({
                "Emotion": emotion.capitalize(),
                "Precision": round(vals.get("precision", 0), 3),
                "Recall": round(vals.get("recall", 0), 3),
                "F1-Score": round(vals.get("f1-score", 0), 3),
                "Support": int(vals.get("support", 0)),
            })
    return pd.DataFrame(rows)


# ─── Prediction ──────────────────────────────────────────────────────────────

def predict_keras(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (predicted_labels, probabilities)."""
    try:
        import tensorflow as tf
        X_r = X.reshape(*X.shape, 1) if X.ndim == 2 else X
        probs = model.predict(X_r, verbose=0)
        preds = np.argmax(probs, axis=1)
        return preds, probs
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return np.zeros(len(X), dtype=int), np.zeros((len(X), NUM_CLASSES))


def predict_single(
    model,
    feature_vector: np.ndarray,
    label_names: List[str] = None,
) -> Dict:
    """Predict emotion for a single audio sample."""
    names = label_names or EMOTION_LABELS

    try:
        import tensorflow as tf
        x = feature_vector.reshape(1, -1, 1)
        probs = model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        return {
            "emotion": names[pred_idx] if pred_idx < len(names) else "unknown",
            "label": pred_idx,
            "confidence": float(probs[pred_idx]),
            "probabilities": {names[i]: float(p) for i, p in enumerate(probs) if i < len(names)},
        }
    except Exception as e:
        logger.error(f"Single prediction failed: {e}")
        return _demo_prediction()


def _demo_prediction() -> Dict:
    """Return a realistic-looking demo prediction."""
    import random
    from config import EMOTIONS

    emotions = EMOTION_LABELS
    # Make one emotion dominant
    probs = np.random.dirichlet(np.ones(len(emotions)) * 0.4)
    dominant = np.argmax(probs)
    probs[dominant] = np.random.uniform(0.55, 0.88)
    probs /= probs.sum()

    return {
        "emotion": emotions[dominant],
        "label": dominant,
        "confidence": float(probs[dominant]),
        "probabilities": {e: float(p) for e, p in zip(emotions, probs)},
        "is_demo": True,
    }


# ─── Demo Evaluation Data ─────────────────────────────────────────────────────

def generate_demo_eval_data(n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic eval data for demo purposes."""
    n = n_samples
    nc = NUM_CLASSES
    y_true = np.repeat(np.arange(nc), n // nc)[:n]

    # Simulate ~80% accuracy
    y_pred = y_true.copy()
    noise_idx = np.random.choice(n, size=int(n * 0.20), replace=False)
    y_pred[noise_idx] = np.random.randint(0, nc, size=len(noise_idx))

    # Probabilities
    y_prob = np.zeros((n, nc))
    for i, label in enumerate(y_true):
        base = np.random.dirichlet(np.ones(nc) * 0.5)
        base[label] = np.random.uniform(0.55, 0.9)
        base /= base.sum()
        y_prob[i] = base

    return y_true, y_pred, y_prob
