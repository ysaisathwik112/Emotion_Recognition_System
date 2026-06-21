"""
Configuration file for Speech Emotion Recognition Platform
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "reports"

for d in [DATASETS_DIR, MODELS_DIR, SAVED_MODELS_DIR, ASSETS_DIR, REPORTS_DIR]:
    d.mkdir(exist_ok=True)

# ─── Audio Settings ───────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION = 3.0          # seconds
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 40
N_CHROMA = 12

SUPPORTED_FORMATS = [".wav", ".mp3", ".ogg", ".flac"]

# ─── Emotions ─────────────────────────────────────────────────────────────────
EMOTIONS = {
    "neutral":  {"label": 0, "color": "#6B7280", "emoji": "😐"},
    "calm":     {"label": 1, "color": "#10B981", "emoji": "😌"},
    "happy":    {"label": 2, "color": "#F59E0B", "emoji": "😊"},
    "sad":      {"label": 3, "color": "#3B82F6", "emoji": "😢"},
    "angry":    {"label": 4, "color": "#EF4444", "emoji": "😠"},
    "fearful":  {"label": 5, "color": "#8B5CF6", "emoji": "😨"},
    "disgust":  {"label": 6, "color": "#84CC16", "emoji": "🤢"},
    "surprise": {"label": 7, "color": "#F97316", "emoji": "😲"},
}

EMOTION_LABELS = list(EMOTIONS.keys())
NUM_CLASSES = len(EMOTIONS)

# ─── Model Configs ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "CNN": {
        "filters": [64, 128, 256],
        "kernel_sizes": [3, 3, 3],
        "dropout": 0.3,
        "dense_units": [256, 128],
    },
    "LSTM": {
        "lstm_units": [128, 64],
        "dropout": 0.3,
        "dense_units": [128],
        "bidirectional": True,
        "attention": True,
    },
    "CNN-LSTM": {
        "cnn_filters": [64, 128],
        "lstm_units": 128,
        "dropout": 0.3,
        "dense_units": [128],
    },
    "Transformer": {
        "model_name": "facebook/wav2vec2-base",
        "num_labels": NUM_CLASSES,
    },
}

# ─── Training ─────────────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "val_split": 0.2,
    "test_split": 0.1,
    "early_stopping_patience": 10,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
}

# ─── UI Theme ─────────────────────────────────────────────────────────────────
THEME = {
    "primary":      "#1A5E3A",   # Rich Forest Green
    "secondary":    "#2D8653",   # Emerald
    "accent":       "#4CAF80",   # Sage Green
    "background":   "#F9F6F0",   # Ivory
    "surface":      "#FFFFFF",
    "surface_alt":  "#F0EDE6",   # Sand / Beige
    "text_primary": "#1C2B2B",   # Dark Slate
    "text_secondary":"#4A5568",  # Charcoal
    "text_muted":   "#718096",
    "border":       "#D4CFC7",
    "success":      "#16A34A",
    "warning":      "#D97706",
    "error":        "#DC2626",
    "info":         "#2D6A4F",
}

# ─── Dataset Configs ──────────────────────────────────────────────────────────
DATASET_CONFIGS = {
    "RAVDESS": {
        "emotion_map": {
            "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
            "05": "angry",   "06": "fearful", "07": "disgust", "08": "surprise",
        },
        "file_pattern": "*.wav",
        "filename_parser": "ravdess",
    },
    "TESS": {
        "emotion_map": {
            "neutral": "neutral", "happy": "happy", "sad": "sad",
            "angry": "angry",     "fear": "fearful", "disgust": "disgust",
            "ps": "surprise",
        },
        "file_pattern": "*.wav",
        "filename_parser": "tess",
    },
    "EMO-DB": {
        "emotion_map": {
            "W": "angry", "L": "calm",   "E": "disgust", "A": "fearful",
            "F": "happy", "T": "sad",    "N": "neutral",
        },
        "file_pattern": "*.wav",
        "filename_parser": "emodb",
    },
}

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
