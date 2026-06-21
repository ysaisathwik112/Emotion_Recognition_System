"""
Utility helpers for the SER platform.
"""

import logging
import time
import io
import base64
from pathlib import Path
from functools import wraps
from typing import Optional

import numpy as np
import pandas as pd

from config import LOG_LEVEL, LOG_FORMAT


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
    return logging.getLogger(name)


logger = get_logger(__name__)


# ─── Decorators ───────────────────────────────────────────────────────────────

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def safe_execute(default=None):
    """Decorator: return `default` on any exception instead of crashing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.error(f"{func.__name__} failed: {exc}", exc_info=True)
                return default
        return wrapper
    return decorator


# ─── Audio helpers ────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def bytes_to_audio_b64(audio_bytes: bytes, fmt: str = "wav") -> str:
    b64 = base64.b64encode(audio_bytes).decode()
    return f"data:audio/{fmt};base64,{b64}"


def audio_file_to_bytes(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ─── Numeric / Array helpers ──────────────────────────────────────────────────

def normalize_array(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), mode="constant")


# ─── DataFrame helpers ────────────────────────────────────────────────────────

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()


# ─── Color / UI helpers ───────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def emotion_badge_html(emotion: str, confidence: float, color: str, emoji: str) -> str:
    return (
        f'<div style="display:inline-flex;align-items:center;gap:8px;'
        f'background:{hex_to_rgba(color,0.12)};border:1.5px solid {color};'
        f'border-radius:8px;padding:8px 16px;">'
        f'<span style="font-size:1.6rem">{emoji}</span>'
        f'<div>'
        f'<div style="font-weight:700;font-size:1rem;color:{color}">{emotion.upper()}</div>'
        f'<div style="font-size:0.8rem;color:#6B7280">{confidence*100:.1f}% confidence</div>'
        f'</div></div>'
    )


# ─── File helpers ─────────────────────────────────────────────────────────────

def get_file_size_str(path: str | Path) -> str:
    size = Path(path).stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def list_audio_files(directory: str | Path, patterns=("*.wav", "*.mp3", "*.ogg", "*.flac")) -> list:
    root = Path(directory)
    files = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(files)


# ─── Streamlit session helpers ────────────────────────────────────────────────

def ensure_session_key(st, key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
