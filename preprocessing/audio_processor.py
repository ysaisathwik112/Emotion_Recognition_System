"""
Audio Preprocessing Pipeline
Handles loading, resampling, noise reduction, VAD, normalization, augmentation.
"""

import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Optional heavy imports — graceful fallback
try:
    import librosa
    import librosa.effects
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa not available – some features degraded")

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


from config import SAMPLE_RATE, DURATION


# ─── Core loader ─────────────────────────────────────────────────────────────

def load_audio(
    source,                          # path str/Path or bytes
    target_sr: int = SAMPLE_RATE,
    duration: Optional[float] = DURATION,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from a file path or raw bytes.
    Returns (waveform: float32 ndarray, sample_rate: int).
    """
    if HAS_LIBROSA:
        if isinstance(source, (str, Path)):
            y, sr = librosa.load(str(source), sr=target_sr, duration=duration, mono=mono)
        else:
            buf = io.BytesIO(source)
            y, sr = librosa.load(buf, sr=target_sr, duration=duration, mono=mono)
    else:
        # Fallback via soundfile
        if isinstance(source, (str, Path)):
            y, sr = sf.read(str(source), dtype="float32", always_2d=False)
        else:
            buf = io.BytesIO(source)
            y, sr = sf.read(buf, dtype="float32", always_2d=False)
        if mono and y.ndim > 1:
            y = y.mean(axis=1)
        if sr != target_sr and HAS_LIBROSA:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    # Ensure float32
    y = y.astype(np.float32)

    # Pad or truncate
    if duration is not None:
        target_len = int(target_sr * duration)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

    return y, sr


# ─── Resampling ───────────────────────────────────────────────────────────────

def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    if HAS_LIBROSA:
        return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    # Simple linear interpolation fallback
    ratio = target_sr / orig_sr
    new_len = int(len(y) * ratio)
    return np.interp(
        np.linspace(0, len(y) - 1, new_len),
        np.arange(len(y)),
        y,
    ).astype(np.float32)


# ─── Noise Reduction ──────────────────────────────────────────────────────────

def reduce_noise(y: np.ndarray, sr: int) -> np.ndarray:
    """Simple spectral subtraction noise reduction."""
    try:
        import noisereduce as nr
        return nr.reduce_noise(y=y, sr=sr)
    except ImportError:
        pass

    if HAS_SCIPY:
        # High-pass filter at 80 Hz to remove low-frequency noise
        b, a = butter(5, 80.0 / (sr / 2), btype="high")
        return filtfilt(b, a, y).astype(np.float32)

    return y  # passthrough


# ─── Silence Removal / VAD ────────────────────────────────────────────────────

def remove_silence(
    y: np.ndarray,
    sr: int,
    top_db: float = 25.0,
    min_silence_ms: float = 200.0,
) -> np.ndarray:
    if not HAS_LIBROSA:
        return y

    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y

    # Minimum gap to actually merge
    min_gap = int(sr * min_silence_ms / 1000)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    chunks = [y[s:e] for s, e in merged]
    return np.concatenate(chunks) if chunks else y


def voice_activity_detection(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    threshold_db: float = -30.0,
) -> np.ndarray:
    """Return boolean mask: True where voice is active."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(y) // frame_len
    mask = np.zeros(len(y), dtype=bool)

    for i in range(n_frames):
        start = i * frame_len
        frame = y[start : start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        if rms_db > threshold_db:
            mask[start : start + frame_len] = True

    return mask


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_audio(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(y ** 2)) + 1e-10
    return (y * target_rms / rms).astype(np.float32)


def peak_normalize(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-10
    return (y / peak).astype(np.float32)


# ─── Signal Enhancement ───────────────────────────────────────────────────────

def preemphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - coef * y[:-1]).astype(np.float32)


def enhance_signal(y: np.ndarray, sr: int) -> np.ndarray:
    y = preemphasis(y)
    y = normalize_audio(y)
    return y


# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment_time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    if not HAS_LIBROSA:
        return y
    return librosa.effects.time_stretch(y, rate=rate)


def augment_pitch_shift(y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    if not HAS_LIBROSA:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def augment_add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y)).astype(np.float32)
    return (y + noise_factor * noise).astype(np.float32)


def augment_shift(y: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift).astype(np.float32)


def get_augmented_samples(
    y: np.ndarray,
    sr: int,
    techniques: list | None = None,
) -> dict:
    if techniques is None:
        techniques = ["noise", "stretch", "pitch", "shift"]

    results = {"original": y}
    try:
        if "noise" in techniques:
            results["noise"] = augment_add_noise(y)
        if "stretch" in techniques:
            results["stretch"] = augment_time_stretch(y, rate=1.1)
        if "pitch" in techniques:
            results["pitch"] = augment_pitch_shift(y, sr, n_steps=2)
        if "shift" in techniques:
            results["shift"] = augment_shift(y)
    except Exception as e:
        logger.warning(f"Augmentation failed: {e}")
    return results


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def full_preprocess_pipeline(
    source,
    target_sr: int = SAMPLE_RATE,
    duration: float = DURATION,
    denoise: bool = True,
    remove_sil: bool = True,
    enhance: bool = True,
) -> Tuple[np.ndarray, int, dict]:
    """
    Run the complete preprocessing pipeline.
    Returns (processed_audio, sample_rate, metadata_dict).
    """
    meta = {}

    y, sr = load_audio(source, target_sr=target_sr, duration=duration)
    meta["original_length"] = len(y)
    meta["sample_rate"] = sr

    if denoise:
        y = reduce_noise(y, sr)

    if remove_sil:
        y_trimmed = remove_silence(y, sr)
        meta["silence_removed_samples"] = len(y) - len(y_trimmed)
        y = y_trimmed

    if enhance:
        y = enhance_signal(y, sr)

    # Final pad / truncate to fixed length
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    else:
        y = y[:target_len]

    meta["final_length"] = len(y)
    meta["duration_sec"] = len(y) / sr

    return y, sr, meta


# ─── Audio info ───────────────────────────────────────────────────────────────

def get_audio_info(y: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    return {
        "duration_sec": len(y) / sr,
        "sample_rate": sr,
        "n_samples": len(y),
        "rms_energy": rms,
        "peak_amplitude": peak,
        "snr_db": 20 * np.log10(rms / (np.std(y) + 1e-10) + 1e-10),
    }


# ─── Convert to bytes ─────────────────────────────────────────────────────────

def audio_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
