"""
Dataset integration: RAVDESS, TESS, EMO-DB loaders and validators.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATASET_CONFIGS, EMOTION_LABELS

logger = logging.getLogger(__name__)


# ─── Filename Parsers ─────────────────────────────────────────────────────────

def parse_ravdess(filepath: Path) -> Optional[str]:
    """
    RAVDESS filename: 03-01-06-01-02-01-12.wav
    Field [2] (index 2) = emotion code 01-08
    """
    name = filepath.stem
    parts = name.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return DATASET_CONFIGS["RAVDESS"]["emotion_map"].get(emotion_code)


def parse_tess(filepath: Path) -> Optional[str]:
    """
    TESS filename: YAF_back_angry.wav  or  OAF_back_sad.wav
    Emotion is the last underscore-separated token.
    """
    name = filepath.stem.lower()
    parts = name.split("_")
    if not parts:
        return None
    emotion_word = parts[-1]
    return DATASET_CONFIGS["TESS"]["emotion_map"].get(emotion_word)


def parse_emodb(filepath: Path) -> Optional[str]:
    """
    EMO-DB filename: 03a01Fa.wav
    The 6th character (index 5) is the emotion code.
    """
    name = filepath.stem
    if len(name) < 6:
        return None
    emotion_code = name[5].upper()
    return DATASET_CONFIGS["EMO-DB"]["emotion_map"].get(emotion_code)


PARSERS = {
    "RAVDESS": parse_ravdess,
    "TESS": parse_tess,
    "EMO-DB": parse_emodb,
}


# ─── Loader ───────────────────────────────────────────────────────────────────

def load_dataset_metadata(
    root_dir: str | Path,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Scan a dataset directory and return a DataFrame with columns:
    filepath, emotion, label, dataset.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    cfg = DATASET_CONFIGS.get(dataset_name)
    if cfg is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS)}")

    parser = PARSERS[dataset_name]

    records = []
    for fp in root.rglob(cfg["file_pattern"]):
        emotion = parser(fp)
        if emotion and emotion in EMOTION_LABELS:
            records.append({
                "filepath": str(fp),
                "filename": fp.name,
                "emotion": emotion,
                "label": EMOTION_LABELS.index(emotion),
                "dataset": dataset_name,
            })

    if not records:
        logger.warning(f"No valid files found in {root} for dataset {dataset_name}")
        return pd.DataFrame(columns=["filepath", "filename", "emotion", "label", "dataset"])

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} samples from {dataset_name}")
    return df


def load_multiple_datasets(dataset_dirs: Dict[str, str]) -> pd.DataFrame:
    """Load and merge multiple datasets. dataset_dirs = {name: path}."""
    dfs = []
    for name, path in dataset_dirs.items():
        try:
            dfs.append(load_dataset_metadata(path, name))
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ─── Statistics & Validation ──────────────────────────────────────────────────

def get_dataset_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    stats = {
        "total_samples": len(df),
        "num_emotions": df["emotion"].nunique(),
        "emotions_present": sorted(df["emotion"].unique().tolist()),
        "class_distribution": df["emotion"].value_counts().to_dict(),
        "datasets": df["dataset"].value_counts().to_dict() if "dataset" in df.columns else {},
        "class_balance_ratio": (
            df["emotion"].value_counts().min() / df["emotion"].value_counts().max()
        ) if len(df) > 0 else 0,
    }
    return stats


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    issues = []

    if df.empty:
        return False, ["Dataset is empty"]

    # Missing emotions
    present = set(df["emotion"].unique())
    missing = set(EMOTION_LABELS) - present
    if missing:
        issues.append(f"Missing emotion classes: {', '.join(sorted(missing))}")

    # Imbalance check
    counts = df["emotion"].value_counts()
    if len(counts) > 1:
        ratio = counts.min() / counts.max()
        if ratio < 0.3:
            issues.append(f"Class imbalance detected (min/max ratio = {ratio:.2f})")

    # File existence spot-check (max 50 files)
    sample = df.sample(min(50, len(df)))
    missing_files = [r for r in sample["filepath"] if not Path(r).exists()]
    if missing_files:
        issues.append(f"{len(missing_files)} sampled files not found on disk")

    return len(issues) == 0, issues


# ─── Feature extraction over a dataset ───────────────────────────────────────

def extract_dataset_features(
    df: pd.DataFrame,
    progress_cb=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature vectors for each row in df.
    Returns (X: (N, F), y: (N,)).
    progress_cb: callable(current, total) for Streamlit progress.
    """
    from preprocessing.audio_processor import full_preprocess_pipeline
    from feature_engineering.feature_extractor import extract_feature_vector

    X, y = [], []
    total = len(df)

    for i, row in df.iterrows():
        try:
            audio, sr, _ = full_preprocess_pipeline(row["filepath"])
            vec = extract_feature_vector(audio, sr)
            X.append(vec)
            y.append(row["label"])
        except Exception as e:
            logger.warning(f"Skipping {row['filepath']}: {e}")

        if progress_cb and (i % 10 == 0):
            progress_cb(len(X), total)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
