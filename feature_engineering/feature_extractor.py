"""
Feature Extraction Engine
Extracts MFCC, spectral, energy, chroma, mel, advanced features.
"""

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa
    import librosa.feature
    import librosa.effects
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.error("librosa missing – feature extraction will fail")

from config import SAMPLE_RATE, N_MFCC, N_MELS, N_FFT, HOP_LENGTH, N_CHROMA


# ─── Individual Feature Functions ─────────────────────────────────────────────

def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return mfcc  # (n_mfcc, T)


def extract_delta_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return librosa.feature.delta(mfcc)


def extract_delta2_mfcc(mfcc: np.ndarray) -> np.ndarray:
    return librosa.feature.delta(mfcc, order=2)


def extract_spectral_centroid(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)


def extract_spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)


def extract_spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)


def extract_spectral_rolloff(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)


def extract_rms(y: np.ndarray) -> np.ndarray:
    return librosa.feature.rms(y=y, hop_length=HOP_LENGTH)


def extract_zcr(y: np.ndarray) -> np.ndarray:
    return librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)


def extract_chroma(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH)


def extract_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


def extract_tonnetz(y: np.ndarray, sr: int) -> np.ndarray:
    y_harm = librosa.effects.harmonic(y)
    return librosa.feature.tonnetz(y=y_harm, sr=sr)


def extract_pitch(y: np.ndarray, sr: int) -> np.ndarray:
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, hop_length=HOP_LENGTH,
    )
    f0 = np.nan_to_num(f0)
    return f0.reshape(1, -1)


# ─── Aggregate stat reduction ─────────────────────────────────────────────────

def _stat_reduce(feat: np.ndarray) -> np.ndarray:
    """Flatten to (mean, std, min, max) per feature row."""
    return np.concatenate([
        feat.mean(axis=1),
        feat.std(axis=1),
        feat.min(axis=1),
        feat.max(axis=1),
    ])


# ─── Master extraction ────────────────────────────────────────────────────────

def extract_all_features(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    as_vector: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Extract the full feature set.
    Returns dict of raw 2D feature arrays.
    If as_vector=True, each value is 1D stats-reduced vector.
    """
    if not HAS_LIBROSA:
        raise RuntimeError("librosa is required for feature extraction")

    features: Dict[str, np.ndarray] = {}

    # MFCC family
    mfcc = extract_mfcc(y, sr)
    features["mfcc"]         = mfcc
    features["delta_mfcc"]   = extract_delta_mfcc(mfcc)
    features["delta2_mfcc"]  = extract_delta2_mfcc(mfcc)

    # Spectral
    features["spectral_centroid"]   = extract_spectral_centroid(y, sr)
    features["spectral_bandwidth"]  = extract_spectral_bandwidth(y, sr)
    features["spectral_contrast"]   = extract_spectral_contrast(y, sr)
    features["spectral_rolloff"]    = extract_spectral_rolloff(y, sr)

    # Energy
    features["rms"] = extract_rms(y)
    features["zcr"] = extract_zcr(y)

    # Frequency
    features["chroma"] = extract_chroma(y, sr)
    features["mel_spectrogram"] = extract_mel_spectrogram(y, sr)

    # Advanced
    try:
        features["tonnetz"] = extract_tonnetz(y, sr)
    except Exception:
        features["tonnetz"] = np.zeros((6, 1))

    try:
        features["pitch"] = extract_pitch(y, sr)
    except Exception:
        features["pitch"] = np.zeros((1, 1))

    if as_vector:
        # Exclude mel_spectrogram from flat vector (too large; used only for viz)
        skip = {"mel_spectrogram"}
        vec = np.concatenate([
            _stat_reduce(v) for k, v in features.items() if k not in skip
        ])
        features["feature_vector"] = vec.astype(np.float32)

    return features


def extract_feature_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Convenience: returns just the flat feature vector."""
    feats = extract_all_features(y, sr, as_vector=True)
    return feats["feature_vector"]


def feature_vector_size() -> int:
    """Return the expected feature vector length."""
    # MFCC(40) + dMFCC(40) + d2MFCC(40): 3×40×4 = 480
    # cent(1)+bw(1)+contrast(7)+rolloff(1) = 10 × 4 = 40
    # rms(1)+zcr(1) = 2×4 = 8
    # chroma(12)×4 = 48
    # tonnetz(6)×4 = 24
    # pitch(1)×4 = 4
    # Total ≈ 604
    return 604
