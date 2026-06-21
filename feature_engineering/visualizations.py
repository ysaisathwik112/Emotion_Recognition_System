"""
Feature visualization utilities (Plotly-based).
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import THEME, SAMPLE_RATE, HOP_LENGTH

PLOT_BG = THEME["surface"]
GRID_COLOR = THEME["border"]
FONT_COLOR = THEME["text_secondary"]
ACCENT = THEME["secondary"]
PRIMARY = THEME["primary"]

LAYOUT_BASE = dict(
    paper_bgcolor=PLOT_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="Inter, sans-serif", color=FONT_COLOR, size=11),
    margin=dict(l=48, r=20, t=40, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=GRID_COLOR, showgrid=True),
)


def _apply_base(fig):
    fig.update_layout(**LAYOUT_BASE)
    return fig


# ─── Waveform ─────────────────────────────────────────────────────────────────

def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE, title: str = "Waveform") -> go.Figure:
    times = np.linspace(0, len(y) / sr, len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=y,
        mode="lines",
        line=dict(color=PRIMARY, width=1),
        name="Amplitude",
        fill="tozeroy",
        fillcolor=f"rgba(26,94,58,0.08)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=FONT_COLOR)),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=220,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── Spectrogram ──────────────────────────────────────────────────────────────

def plot_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> go.Figure:
    try:
        import librosa
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    except Exception:
        D = np.zeros((128, 100))

    fig = go.Figure(go.Heatmap(
        z=D, colorscale="Viridis", showscale=True,
        colorbar=dict(title="dB", thickness=12, len=0.8),
    ))
    fig.update_layout(
        title="Spectrogram",
        xaxis_title="Frame", yaxis_title="Frequency Bin",
        height=260,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── Mel Spectrogram ──────────────────────────────────────────────────────────

def plot_mel_spectrogram(mel_db: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=mel_db, colorscale="RdYlGn", showscale=True,
        colorbar=dict(title="dB", thickness=12, len=0.8),
    ))
    fig.update_layout(
        title="Mel Spectrogram",
        xaxis_title="Frame", yaxis_title="Mel Band",
        height=260,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── MFCC Heatmap ────────────────────────────────────────────────────────────

def plot_mfcc(mfcc: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=mfcc, colorscale="RdBu_r", showscale=True,
        colorbar=dict(title="Value", thickness=12, len=0.8),
    ))
    fig.update_layout(
        title="MFCC Coefficients",
        xaxis_title="Frame", yaxis_title="MFCC Index",
        height=260,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── Feature Stats Bar ────────────────────────────────────────────────────────

def plot_feature_stats(feature_vector: np.ndarray) -> go.Figure:
    n = min(40, len(feature_vector))
    x = list(range(n))
    y = feature_vector[:n].tolist()
    colors = [ACCENT if v >= 0 else "#EF4444" for v in y]

    fig = go.Figure(go.Bar(
        x=x, y=y, marker_color=colors, name="Feature Value",
    ))
    fig.update_layout(
        title="Feature Vector (first 40 dims)",
        xaxis_title="Feature Index", yaxis_title="Value",
        height=240,
        showlegend=False,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── Emotion Confidence Bar ───────────────────────────────────────────────────

def plot_confidence_bars(emotion_probs: dict) -> go.Figure:
    """emotion_probs: {emotion: probability}"""
    from config import EMOTIONS

    labels = list(emotion_probs.keys())
    values = [emotion_probs[e] * 100 for e in labels]
    colors = [EMOTIONS.get(e, {}).get("color", ACCENT) for e in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    max_val = max(values) if values else 100
    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis=dict(title="Confidence (%)", range=[0, min(max_val * 1.25, 105)],
                   gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
        height=320,
        showlegend=False,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


# ─── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, labels: list) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#F9F6F0"], [0.5, ACCENT], [1, PRIMARY]],
        showscale=True,
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=12),
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        height=420,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ─── Training Curves ──────────────────────────────────────────────────────────

def plot_training_curves(history: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Loss", "Accuracy"],
        shared_xaxes=False,
    )
    epochs = list(range(1, len(history.get("loss", [])) + 1))

    if "loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines",
                                 name="Train Loss", line=dict(color=PRIMARY, width=2)), row=1, col=1)
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines",
                                 name="Val Loss", line=dict(color="#D97706", width=2, dash="dash")), row=1, col=1)
    if "accuracy" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"], mode="lines",
                                 name="Train Acc", line=dict(color=ACCENT, width=2)), row=1, col=2)
    if "val_accuracy" in history:
        fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], mode="lines",
                                 name="Val Acc", line=dict(color="#8B5CF6", width=2, dash="dash")), row=1, col=2)

    fig.update_layout(
        height=300,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, sans-serif", color=FONT_COLOR),
        margin=dict(l=48, r=20, t=48, b=40),
        legend=dict(orientation="h", y=-0.15),
    )
    for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{axis: dict(gridcolor=GRID_COLOR)})
    return fig


# ─── Class Distribution ───────────────────────────────────────────────────────

def plot_class_distribution(label_counts: dict) -> go.Figure:
    from config import EMOTIONS

    labels = list(label_counts.keys())
    values = list(label_counts.values())
    colors = [EMOTIONS.get(e, {}).get("color", ACCENT) for e in labels]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Emotion", yaxis_title="Count",
        height=300,
        showlegend=False,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR),
    )
    return fig


# ─── Emotion Timeline ────────────────────────────────────────────────────────

def plot_emotion_timeline(timestamps: list, emotions: list, confidences: list) -> go.Figure:
    from config import EMOTIONS

    colors = [EMOTIONS.get(e, {}).get("color", ACCENT) for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=confidences,
        mode="markers+lines",
        marker=dict(color=colors, size=12, line=dict(width=1, color="white")),
        line=dict(color=GRID_COLOR, width=1),
        text=emotions,
        hovertemplate="<b>%{text}</b><br>Confidence: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Emotion Timeline",
        xaxis_title="Time (s)", yaxis_title="Confidence",
        height=260,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, range=[0, 1.05]),
    )
    return fig


# ─── ROC Curves ──────────────────────────────────────────────────────────────

def plot_roc_curves(fpr_dict: dict, tpr_dict: dict, auc_dict: dict) -> go.Figure:
    from config import EMOTIONS

    fig = go.Figure()
    for emotion, fpr in fpr_dict.items():
        auc = auc_dict.get(emotion, 0)
        color = EMOTIONS.get(emotion, {}).get("color", ACCENT)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr_dict[emotion],
            mode="lines",
            name=f"{emotion.capitalize()} (AUC={auc:.2f})",
            line=dict(color=color, width=2),
        ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color=GRID_COLOR), showlegend=False))
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        legend=dict(orientation="v", x=0.75, y=0.05),
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        xaxis=dict(gridcolor=GRID_COLOR, range=[0, 1]),
        yaxis=dict(gridcolor=GRID_COLOR, range=[0, 1.02]),
    )
    return fig
