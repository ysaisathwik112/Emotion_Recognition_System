"""
Model training pipeline with cross-validation, early stopping, checkpointing.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import TRAINING_CONFIG, SAVED_MODELS_DIR, EMOTION_LABELS

logger = logging.getLogger(__name__)

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False


# ─── Data Splitting ───────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple:
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required")

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test) -> Tuple:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


# ─── Training ─────────────────────────────────────────────────────────────────

def train_keras_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict = None,
    checkpoint_name: str = "best_model",
) -> Dict:
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")

    cfg = config or TRAINING_CONFIG
    checkpoint_path = str(SAVED_MODELS_DIR / f"{checkpoint_name}.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["early_stopping_patience"],
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy",
            save_best_only=True, verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["reduce_lr_factor"],
            patience=cfg["reduce_lr_patience"], min_lr=1e-6, verbose=0,
        ),
    ]

    # Reshape for CNN/LSTM if needed
    if len(X_train.shape) == 2:
        X_train_r = X_train.reshape(*X_train.shape, 1)
        X_val_r = X_val.reshape(*X_val.shape, 1)
    else:
        X_train_r, X_val_r = X_train, X_val

    start = time.time()
    history = model.fit(
        X_train_r, y_train,
        validation_data=(X_val_r, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=0,
    )
    elapsed = time.time() - start

    return {
        "history": history.history,
        "best_val_accuracy": max(history.history.get("val_accuracy", [0])),
        "best_val_loss": min(history.history.get("val_loss", [999])),
        "epochs_run": len(history.history["loss"]),
        "training_time_sec": elapsed,
        "checkpoint_path": checkpoint_path,
    }


def cross_validate_keras(
    model_builder,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    config: dict = None,
) -> Dict:
    if not (HAS_TF and HAS_SKLEARN):
        raise RuntimeError("TensorFlow + scikit-learn required")

    cfg = config or TRAINING_CONFIG
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        model = model_builder()
        X_t, X_v = X[train_idx], X[val_idx]
        y_t, y_v = y[train_idx], y[val_idx]

        X_t_r = X_t.reshape(*X_t.shape, 1)
        X_v_r = X_v.reshape(*X_v.shape, 1)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ]

        hist = model.fit(
            X_t_r, y_t, validation_data=(X_v_r, y_v),
            epochs=min(cfg["epochs"], 20),
            batch_size=cfg["batch_size"],
            callbacks=callbacks, verbose=0,
        )
        val_acc = max(hist.history.get("val_accuracy", [0]))
        scores.append(val_acc)
        logger.info(f"Fold {fold+1}/{n_splits}: val_accuracy={val_acc:.4f}")

    return {
        "fold_scores": scores,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
    }


# ─── Saving / Loading ─────────────────────────────────────────────────────────

def save_model(model, name: str) -> str:
    path = str(SAVED_MODELS_DIR / f"{name}.keras")
    model.save(path)
    logger.info(f"Model saved to {path}")
    return path


def load_model(name: str):
    if not HAS_TF:
        raise RuntimeError("TensorFlow required")
    path = SAVED_MODELS_DIR / f"{name}.keras"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return keras.models.load_model(str(path))


def list_saved_models() -> list:
    return [p.stem for p in SAVED_MODELS_DIR.glob("*.keras")]


# ─── Demo Training (synthetic data) ──────────────────────────────────────────

def generate_demo_training_result() -> Dict:
    """
    Return a plausible training result without actual training,
    used when no real dataset is loaded.
    """
    import random
    n_epochs = random.randint(20, 40)
    base_loss = 2.0
    base_acc = 0.12

    loss, val_loss, acc, val_acc = [], [], [], []
    for i in range(n_epochs):
        decay = np.exp(-i * 0.12)
        noise = np.random.normal(0, 0.015)
        loss.append(float(base_loss * decay + 0.22 + noise))
        val_loss.append(float(base_loss * decay + 0.28 + noise * 1.2))
        a = min(0.92, base_acc + (0.92 - base_acc) * (1 - decay) + noise * 0.5)
        acc.append(float(a))
        va = min(0.87, base_acc + (0.87 - base_acc) * (1 - decay) + noise * 0.5)
        val_acc.append(float(va))

    return {
        "history": {
            "loss": loss, "val_loss": val_loss,
            "accuracy": acc, "val_accuracy": val_acc,
        },
        "best_val_accuracy": max(val_acc),
        "best_val_loss": min(val_loss),
        "epochs_run": n_epochs,
        "training_time_sec": random.uniform(45, 180),
        "checkpoint_path": "demo_model.keras",
        "is_demo": True,
    }
