"""
Explainable AI: SHAP analysis, feature importance, attention visualization.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import THEME, EMOTION_LABELS

logger = logging.getLogger(__name__)

PLOT_BG = THEME["surface"]
GRID_COLOR = THEME["border"]
FONT_COLOR = THEME["text_secondary"]
PRIMARY = THEME["primary"]
ACCENT = THEME["secondary"]

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─── Feature Names ────────────────────────────────────────────────────────────

def build_feature_names() -> List[str]:
    """Build human-readable names for the feature vector."""
    names = []
    stats = ["mean", "std", "min", "max"]

    # MFCC × 40 × 4 stats
    for kind in ["MFCC", "Delta_MFCC", "Delta2_MFCC"]:
        for i in range(40):
            for s in stats:
                names.append(f"{kind}_{i+1}_{s}")

    # Spectral × 4
    for feat in ["Centroid", "Bandwidth"]:
        for s in stats:
            names.append(f"Spectral_{feat}_{s}")

    # Spectral Contrast × 7 × 4
    for i in range(7):
        for s in stats:
            names.append(f"SpectralContrast_{i+1}_{s}")

    # Rolloff × 4
    for s in stats:
        names.append(f"SpectralRolloff_{s}")

    # RMS × 4
    for s in stats:
        names.append(f"RMS_{s}")

    # ZCR × 4
    for s in stats:
        names.append(f"ZCR_{s}")

    # Chroma × 12 × 4
    for i in range(12):
        for s in stats:
            names.append(f"Chroma_{i+1}_{s}")

    # Tonnetz × 6 × 4
    for i in range(6):
        for s in stats:
            names.append(f"Tonnetz_{i+1}_{s}")

    # Pitch × 4
    for s in stats:
        names.append(f"Pitch_{s}")

    return names


FEATURE_NAMES = build_feature_names()


# ─── Gradient-based Feature Importance (TF) ──────────────────────────────────

def gradient_feature_importance(
    model,
    feature_vector: np.ndarray,
    top_k: int = 20,
) -> Dict:
    try:
        import tensorflow as tf
        x = tf.Variable(feature_vector.reshape(1, -1, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            preds = model(x)
            pred_class = tf.argmax(preds[0])
            score = preds[0, pred_class]
        grads = tape.gradient(score, x)
        importance = np.abs(grads.numpy()).flatten()[:len(feature_vector)]
    except Exception as e:
        logger.warning(f"Gradient importance failed: {e}")
        importance = np.abs(feature_vector)

    # Normalize
    importance = importance / (importance.max() + 1e-10)
    top_indices = np.argsort(importance)[-top_k:][::-1]

    return {
        "importance": importance,
        "top_indices": top_indices.tolist(),
        "top_names": [FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}" for i in top_indices],
        "top_values": importance[top_indices].tolist(),
    }


# ─── SHAP Analysis ───────────────────────────────────────────────────────────

def shap_analysis(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_background: int = 50,
) -> Optional[Dict]:
    if not HAS_SHAP:
        return None

    try:
        background = X_background[:n_background]
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x.reshape(len(x), -1, 1), verbose=0),
            background,
        )
        shap_values = explainer.shap_values(X_explain[:5])
        return {"shap_values": shap_values, "background": background}
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        return None


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_feature_importance(importance_dict: Dict, title: str = "Feature Importance") -> go.Figure:
    names = importance_dict["top_names"][:15]
    values = importance_dict["top_values"][:15]

    # Short names for display
    short_names = [n.replace("_mean", "").replace("_std", " σ") for n in names]
    colors = [f"rgba(45,134,83,{0.4 + 0.6*v})" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=short_names[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score", yaxis_title="",
        height=420,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=11),
        margin=dict(l=160, r=60, t=40, b=40),
        xaxis=dict(gridcolor=GRID_COLOR, range=[0, max(values) * 1.2 + 0.05]),
        yaxis=dict(gridcolor=GRID_COLOR),
        showlegend=False,
    )
    return fig


def plot_feature_group_importance(importance: np.ndarray) -> go.Figure:
    """Aggregate importance by feature group."""
    groups = {
        "MFCC": (0, 160),
        "Delta MFCC": (160, 320),
        "Delta² MFCC": (320, 480),
        "Spectral": (480, 520),
        "Energy": (520, 528),
        "Chroma": (528, 576),
        "Tonnetz": (576, 600),
        "Pitch": (600, 604),
    }

    group_imp = {}
    total = importance.sum() + 1e-10
    for name, (start, end) in groups.items():
        end = min(end, len(importance))
        if start < end:
            group_imp[name] = float(importance[start:end].sum() / total * 100)

    labels = list(group_imp.keys())
    values = list(group_imp.values())
    colors_list = [
        PRIMARY, ACCENT, "#4CAF80", "#D97706",
        "#8B5CF6", "#F59E0B", "#EF4444", "#3B82F6",
    ]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors_list[:len(labels)],
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Feature Group Contribution",
        height=380,
        paper_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR),
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(orientation="h", y=-0.08),
        annotations=[dict(text="Groups", x=0.5, y=0.5, font_size=13, showarrow=False)],
    )
    return fig


def explain_prediction(
    model,
    feature_vector: np.ndarray,
    pred_result: Dict,
) -> Dict:
    """Generate a human-readable explanation of the prediction."""
    importance = gradient_feature_importance(model, feature_vector)

    top_group_contributions = {}
    imp = importance["importance"]
    groups = {
        "MFCC features": (0, 480),
        "Spectral features": (480, 528),
        "Energy features": (528, 576),
        "Chroma & Tonnetz": (576, 604),
    }
    for g, (s, e) in groups.items():
        e = min(e, len(imp))
        top_group_contributions[g] = float(imp[s:e].sum() / (imp.sum() + 1e-10))

    dominant_group = max(top_group_contributions, key=top_group_contributions.get)

    explanation = (
        f"The model predicted **{pred_result['emotion'].capitalize()}** "
        f"with {pred_result['confidence']:.1%} confidence. "
        f"The most influential feature group was **{dominant_group}** "
        f"({top_group_contributions[dominant_group]:.0%} contribution). "
        f"The top feature was **{importance['top_names'][0]}** "
        f"(importance: {importance['top_values'][0]:.3f})."
    )

    return {
        "importance_dict": importance,
        "group_contributions": top_group_contributions,
        "dominant_group": dominant_group,
        "explanation_text": explanation,
    }
