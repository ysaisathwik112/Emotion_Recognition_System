"""
Deep Learning Model Architectures for Speech Emotion Recognition.
CNN, LSTM, CNN-LSTM Hybrid, Transformer (wav2vec2).
"""

import logging
from typing import Optional, Tuple

import numpy as np

from config import NUM_CLASSES, MODEL_CONFIGS

logger = logging.getLogger(__name__)

# ─── TensorFlow / Keras ───────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not available – TF models disabled")

# ─── PyTorch ─────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available – PT models disabled")

# ─── HuggingFace ─────────────────────────────────────────────────────────────
try:
    from transformers import (
        Wav2Vec2ForSequenceClassification,
        Wav2Vec2Config,
        AutoFeatureExtractor,
    )
    HAS_HF = True
except ImportError:
    HAS_HF = False


# ══════════════════════════════════════════════════════════════════════════════
# TensorFlow / Keras Models
# ══════════════════════════════════════════════════════════════════════════════

def build_cnn_model(input_shape: Tuple, num_classes: int = NUM_CLASSES) -> "keras.Model":
    """1D CNN for fixed-length feature vectors."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    cfg = MODEL_CONFIGS["CNN"]
    inp = keras.Input(shape=input_shape, name="features")
    x = layers.Reshape((*input_shape, 1))(inp) if len(input_shape) == 1 else inp

    for filters, ks in zip(cfg["filters"], cfg["kernel_sizes"]):
        x = layers.Conv1D(filters, ks, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(cfg["dropout"])(x)

    x = layers.GlobalAveragePooling1D()(x)
    for units in cfg["dense_units"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(cfg["dropout"])(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = Model(inp, out, name="SER_CNN")
    return model


def build_lstm_model(input_shape: Tuple, num_classes: int = NUM_CLASSES) -> "keras.Model":
    """Bidirectional LSTM with attention."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    cfg = MODEL_CONFIGS["LSTM"]
    inp = keras.Input(shape=input_shape, name="features")
    x = inp

    for i, units in enumerate(cfg["lstm_units"]):
        return_seq = (i < len(cfg["lstm_units"]) - 1) or cfg["attention"]
        lstm_layer = layers.LSTM(units, return_sequences=return_seq, dropout=cfg["dropout"])
        if cfg["bidirectional"]:
            x = layers.Bidirectional(lstm_layer)(x)
        else:
            x = lstm_layer(x)

    if cfg["attention"]:
        # Self-attention
        attn = layers.Dense(1, activation="tanh")(x)
        attn = layers.Flatten()(attn)
        attn = layers.Activation("softmax")(attn)
        attn = layers.RepeatVector(x.shape[-1])(attn)
        attn = layers.Permute([2, 1])(attn)
        x = layers.Multiply()([x, attn])
        x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    for units in cfg["dense_units"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(cfg["dropout"])(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inp, out, name="SER_LSTM")


def build_cnn_lstm_model(input_shape: Tuple, num_classes: int = NUM_CLASSES) -> "keras.Model":
    """CNN feature extractor + LSTM temporal learner."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    cfg = MODEL_CONFIGS["CNN-LSTM"]
    inp = keras.Input(shape=input_shape, name="features")
    x = inp

    for filters in cfg["cnn_filters"]:
        x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

    x = layers.LSTM(cfg["lstm_units"], return_sequences=True, dropout=cfg["dropout"])(x)

    # Attention
    attn = layers.Dense(1, activation="tanh")(x)
    attn = layers.Flatten()(attn)
    attn = layers.Activation("softmax")(attn)
    attn = layers.RepeatVector(cfg["lstm_units"])(attn)
    attn = layers.Permute([2, 1])(attn)
    x = layers.Multiply()([x, attn])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    for units in cfg["dense_units"]:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(cfg["dropout"])(x)

    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return Model(inp, out, name="SER_CNN_LSTM")


def compile_model(model, lr: float = 0.001):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(checkpoint_path: str, patience: int = 10):
    if not HAS_TF:
        return []
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=0,
        ),
        keras.callbacks.CSVLogger(checkpoint_path.replace(".h5", "_log.csv")),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Model
# ══════════════════════════════════════════════════════════════════════════════

if HAS_TORCH:
    class SERTransformerPT(nn.Module):
        """Lightweight Transformer-based classifier for fixed-length feature vectors."""

        def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES,
                     d_model: int = 128, nhead: int = 4, num_layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)          # (B, 1, F)
            x = self.proj(x)                # (B, T, d_model)
            x = self.transformer(x)
            x = self.norm(x.mean(dim=1))    # mean pooling
            return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace Wav2Vec2
# ══════════════════════════════════════════════════════════════════════════════

def build_wav2vec2_classifier(num_labels: int = NUM_CLASSES):
    if not HAS_HF:
        raise RuntimeError("transformers library required")
    config = Wav2Vec2Config.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model = Wav2Vec2ForSequenceClassification(config)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    model_type: str,
    input_shape: Tuple,
    num_classes: int = NUM_CLASSES,
    lr: float = 0.001,
):
    """Build and compile a model by name."""
    if model_type == "CNN":
        m = build_cnn_model(input_shape, num_classes)
    elif model_type == "LSTM":
        m = build_lstm_model(input_shape, num_classes)
    elif model_type == "CNN-LSTM":
        m = build_cnn_lstm_model(input_shape, num_classes)
    elif model_type == "Transformer":
        if HAS_TORCH:
            return SERTransformerPT(input_shape[0], num_classes)
        raise RuntimeError("PyTorch required for Transformer model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return compile_model(m, lr=lr)


def model_summary_str(model) -> str:
    if HAS_TF and isinstance(model, keras.Model):
        lines = []
        model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)
    if HAS_TORCH and isinstance(model, nn.Module):
        return str(model)
    return "Model summary unavailable"
